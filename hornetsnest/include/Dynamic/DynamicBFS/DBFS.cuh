#pragma once

#include "HornetAlg.hpp"
#include <BufferPool.cuh>
#include <Device/Util/Timer.cuh>
#include <LoadBalancing/LogarithmRadixBinning.cuh>
#include <Macros.hpp>
#include <Static/BreadthFirstSearch/TopDown2.cuh>
#include <algorithm>
#include <cstdio>
#include <cuda_device_runtime_api.h>
#include <device_atomic_functions.h>
#include <iterator>
#include <stdio.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

namespace hornets_nest {

using vert_t = int;
using dist_t = int;

using BatchUpdate = hornet::gpu::BatchUpdate<vert_t>;

template <typename HornetGraph>
class DynamicBFS : public StaticAlgorithm<HornetGraph> {
public:
  // Contains useful statistics
  struct Stats {

    // General frontier data
    int frontier_expansions_count;
    int initial_frontier_size;
    int min_frontier_size;
    int max_frontier_size;
    int median_frontier_size;

    // Time measures for dynamic update
    float vertex_update_time;
    float expansion_time;

    float bfs_time;
    int bfs_max_level;

    typename BfsTopDown2<HornetGraph>::Stats bfs_stats;
  };

public:
  DynamicBFS(HornetGraph &graph, HornetGraph &parent_graph);
  virtual ~DynamicBFS();

  void reset() override;
  void run() override;
  void release() override;
  bool validate() override;

  void set_source(vert_t source);
  dist_t get_current_level() const;
  Stats get_stats() const;
  dist_t *get_host_distance_vector() const;

  void set_device_distance_vector(int *distances);

  /// @brief Process the inserted edges to generate an updated
  /// distance array
  void update(BatchUpdate &batch);

  /// @brief Process the removed edges to generate an updatet distance array
  void erase(BatchUpdate &batch, bool verbose);

  /// @brief Process the inserted of removed edges to generate an updated
  /// distance array
  void update_bottom_up(BatchUpdate &batch);

  /// @brief Apply a node relabeling to the stored distance vector
  void relabel(const thrust::device_vector<vert_t> &map);

  /// @brief Return current distance vector
  dist_t *current_distances();
  /// @brief Return writable distance vector
  dist_t *write_distances();

  /// @brief Swap the current distance vector
  void swap_distances();
  /// #brief Copy the current distances into the others
  void sync_distances();

  void swap_and_sync_distances() {
    swap_distances();
    sync_distances();
  }

private:
  HornetGraph &_graph;
  HornetGraph &_parent_graph;

  BufferPool _buffer_pool;
  load_balancing::BinarySearch _load_balancing;
  TwoLevelQueue<vert_t> _frontier;
  Stats _stats{};

  // Used to measure algorithm performance
  timer::Timer<timer::DEVICE> _device_timer;

  int _current_distance_vector{0};
  dist_t *_distances[2] = {nullptr, nullptr};

  vert_t _source{0};
  dist_t _current_level{0};

  /// The last calculated permutation of the graph nodes to increase cache
  /// coherency
  dist_t *_relabeling{nullptr};
};

} // namespace hornets_nest

namespace hornets_nest {

// static const dist_t INF = std::numeric_limits<dist_t>::max() - 1;

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//                          Algorithm Operators
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

struct Reset {
  dist_t *distances;
  OPERATOR(Vertex &vertex) { distances[vertex.id()] = INF; }
};

struct Expand {

  dist_t current_level;
  dist_t *d_distances;
  TwoLevelQueue<vert_t> queue;

  OPERATOR(Vertex &vertex, Edge &edge) {
    // printf("Expanding %d -> %d\n", vertex.id(), edge.dst_id());
    auto dst = edge.dst_id();
    if (d_distances[dst] == INF) {
      if (atomicCAS(d_distances + dst, INF, current_level) == INF) {
        queue.insert(dst);

        /*
        printf("\tAdded new node to frontier and set new level for node %d:
        %d\n", edge.dst_id(), d_distances[dst]);
        */
      }
    }
  }
};

struct PrintNeighbours {
  vert_t target_id;
  OPERATOR(Vertex &vertex, Edge &edge) {
    if (vertex.id() == target_id)
      printf("edge: %d -> %d\n", edge.src_id(), edge.dst_id());
  }
};

// Find the smallest parent of the vertex
struct VertexUpdate {

  dist_t *read_distances;
  dist_t *write_distances;

  TwoLevelQueue<vert_t> frontier;

  OPERATOR(Vertex &vertex, Edge &edge) {
    // printf("Analyzing edge %d[d: %d] -> %d[d: %d] for best father\n",
    //        vertex.id(), read_distances[vertex.id()], edge.dst_id(),
    //        read_distances[edge.dst_id()]);

#if 1
    // WORKING VERSION

    const int distance = read_distances[vertex.id()];
    if (read_distances[edge.dst_id()] < distance - 1) {
      if (atomicMin(&write_distances[vertex.id()],
                    read_distances[edge.dst_id()] + 1) == distance) {
        frontier.insert(vertex.id());
      }

      // printf("\tNew smallest father for %d: %d at distance %d\n",
      //        vertex.id(), edge.dst_id(), read_distances[edge.dst_id()]);
    }
#else
    atomicMin(&distances[vertex.id()], distances[edge.dst_id()] + 1);
#endif
  }
};

struct BatchVertexUpdate {

  dist_t *read_distances, *write_distances;
  TwoLevelQueue<vert_t> frontier;

  OPERATOR(Vertex &src, Vertex &dst) {

    const int src_distance = read_distances[src.id()];
    const int dst_distance = read_distances[dst.id()];

    const int delta = dst_distance - src_distance;
    if (delta > 1) {
      if (atomicMin(&write_distances[dst.id()], src_distance + 1) ==
          dst_distance)
        frontier.insert(dst.id());
    }
  }
};

// Foreach node children overwrite old distance if we are closer
struct DynamicExpandEdge {

  dist_t *read_distances;
  dist_t *write_distances;

  TwoLevelQueue<vert_t> frontier;

  OPERATOR(Vertex &vertex, Edge &edge) {
    // printf("Dynamic expand from %d[%d] -> %d[%d]\n",
    //        vertex.id(), read_distances[vertex.id()], edge.dst_id(),
    //        read_distances[edge.dst_id()]);

#if 1
    // WORKING VERSION

    const int our_dist = read_distances[vertex.id()];
    const int dst_dist = read_distances[edge.dst_id()];

    const int delta = dst_dist - our_dist;
    if (delta > 1) {

      // Modify distance of destination node, if the operation returns our read
      // value then we must add to queue
      if (atomicMin(&write_distances[edge.dst_id()], our_dist + 1) ==
          dst_dist) {
        frontier.insert(edge.dst_id());
      }

      // printf("\tNode %d has been updated to distance %d\n",
      //        edge.dst_id(), our_dist + 1);
    }
#else
    const dist_t old_distance = distances[edge.dst_id()];
    if (atomicMin(&distances[edge.dst_id()], distances[vertex.id()] + 1) ==
        old_distance)
      frontier.insert(edge.dst_id());
#endif
  }
};

struct BUBatchVertexUpdate {
  dist_t *read_distances, *write_distances;
  TwoLevelQueue<vert_t> frontier;

  OPERATOR(Vertex &src, Vertex &dst) {

    const int src_distance = read_distances[src.id()];
    const int dst_distance = read_distances[dst.id()];

    const int delta = dst_distance - src_distance;
    if (delta > 1) {
      atomicMin(&write_distances[dst.id()], src_distance + 1);
    }
  }
};

struct BUDynamicExpandEdge {

  int *work_to_do;
  dist_t *read_distances, *write_distances;

  OPERATOR(Vertex &vertex, Edge &edge) {

    const int our_dist = read_distances[vertex.id()];
    const int src_dist =
        read_distances[edge.dst_id()]; // We are using the parent graph

    const int delta = our_dist - src_dist;
    if (delta > 1) {
      // Get the lowest parent distance
      if (atomicMin(&write_distances[vertex.id()], src_dist + 1) == our_dist)
        *work_to_do = true;
    }
  }
};

//------------------------------------------------------------------------------

template <typename HornetGraph>
DynamicBFS<HornetGraph>::DynamicBFS(HornetGraph &graph,
                                    HornetGraph &parent_graph)
    : StaticAlgorithm<HornetGraph>{graph}, _load_balancing{graph},
      _frontier{graph, 10.0f}, _graph{graph}, _parent_graph{parent_graph} {
  _buffer_pool.allocate(&_distances[0], _graph.nV());
  _buffer_pool.allocate(&_distances[1], _graph.nV());
  _buffer_pool.allocate(&_relabeling, _graph.nV());
  reset();
}

template <typename HornetGraph> DynamicBFS<HornetGraph>::~DynamicBFS() {}

template <typename HornetGraph>
void DynamicBFS<HornetGraph>::set_source(vert_t source) {
  _source = source;
}

template <typename HornetGraph>
dist_t DynamicBFS<HornetGraph>::get_current_level() const {
  return _current_level;
}

template <typename HornetGraph>
void DynamicBFS<HornetGraph>::set_device_distance_vector(int *distances) {
  cudaMemcpy(current_distances(), distances, _graph.nV(),
             cudaMemcpyDeviceToDevice);
}

template <typename HornetGraph>
auto DynamicBFS<HornetGraph>::get_stats() const -> Stats {
  return _stats;
}

template <typename HornetGraph>
dist_t *DynamicBFS<HornetGraph>::get_host_distance_vector() const {
  dist_t *result = new dist_t[_graph.nV()];
  gpu::copyToHost(_distances[_current_distance_vector], _graph.nV(), result);
  return result;
}

template <typename HornetGraph>
dist_t *DynamicBFS<HornetGraph>::current_distances() {
  return _distances[_current_distance_vector];
}

template <typename HornetGraph>
dist_t *DynamicBFS<HornetGraph>::write_distances() {
  return _distances[!_current_distance_vector];
}

template <typename HornetGraph> void DynamicBFS<HornetGraph>::swap_distances() {
  _current_distance_vector = !_current_distance_vector;
}

// Copy the content of the 'from' vector to the 'to' vector
__global__ void sync_distances_kernel(const dist_t *src, dist_t *dst,
                                      int size) {

  int stride = blockDim.x * gridDim.x;
  int start = blockIdx.x * blockDim.x + threadIdx.x;

  for (auto i = start; i < size; i += stride) {
    dst[i] = src[i];
  }
}

template <typename HornetGraph> void DynamicBFS<HornetGraph>::sync_distances() {
  dist_t *src_distances = _distances[_current_distance_vector];
  dist_t *dst_distances = _distances[!_current_distance_vector];

  int block_count = xlib::ceil_div<1024>(_graph.nV());
  sync_distances_kernel<<<block_count, 1024>>>(src_distances, dst_distances,
                                               _graph.nV());
}

template <typename HornetGraph> void DynamicBFS<HornetGraph>::reset() {
  _current_level = 1;
  _frontier.clear();

  dist_t *distances = current_distances();
  forAllnumV(StaticAlgorithm<HornetGraph>::hornet,
             [=] __device__(int i) { distances[i] = INF; });
}

// Normal BFS
template <typename HornetGraph> void DynamicBFS<HornetGraph>::run() {
  _frontier.insert(_source);
  gpu::memsetZero(current_distances() + _source);

  assert(_frontier.size() != 0);
  while (_frontier.size() != 0) {

    forAllEdges(StaticAlgorithm<HornetGraph>::hornet, _frontier,
                Expand{_current_level, current_distances(), _frontier},
                _load_balancing);

    _current_level += 1;
    _frontier.swap();
  }

#if 0
    // Print distance vector
    cudaDeviceSynchronize();
    auto* dbfs_host_distances = new dist_t[_graph.nV()];
    gpu::copyToHost(_distances, _graph.nV(), dbfs_host_distances);
    for (int i = 0; i < _graph.nV(); i++)
        printf("[DBFS] node: %d | dist: %d\n", i, dbfs_host_distances[i]);
    delete[] dbfs_host_distances;
#endif
}

/*
/// Size of the shared memory per block used for the local histogram
#define PARTIAL_HIST_SIZE 4096

/// @brief Calculate a partial histogram of the distances for each block
__global__ void kernel_block_histogram(const dist_t *distances, const int N,
                                       dist_t *result, dist_t min, dist_t max) {

  int grid_size = blockDim.x * gridDim.x;
  int block_size = blockDim.x;
  int block_idx = blockIdx.x;

  // Local histogram for each block
  __shared__ dist_t local_histogram[PARTIAL_HIST_SIZE];
  for (int i = threadIdx.x; i < PARTIAL_HIST_SIZE; i += block_size)
    local_histogram[i] = 0;

  __syncthreads();

  // Accumulate atomically on shared memory
  int global_idx = block_idx * block_size + threadIdx.x;
  for (int i = global_idx; i < N; i += grid_size) {
    const dist_t distance = distances[i];
    if (distance >= min && distance < max)
      atomicAdd(&local_histogram[distance - min], 1);
  }

  __syncthreads();

  // Send local histogram to global memory
  for (int i = threadIdx.x; i < PARTIAL_HIST_SIZE; i += block_size)
    result[block_idx * PARTIAL_HIST_SIZE + i] = local_histogram[i];
}

/// @brief Accumulate all local histograms
__global__ void kernel_accumulate_histogram(const dist_t *blocks,
                                            const int blocks_count,
                                            dist_t *result) {

  int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (global_idx < PARTIAL_HIST_SIZE) {
    dist_t accumulator = 0;
    for (int block_idx = 0; block_idx < blocks_count; block_idx += 1)
      accumulator += blocks[block_idx * PARTIAL_HIST_SIZE + global_idx];
    result[global_idx] = accumulator;
  }
}

/// @brief Distribute all nodes in their correct position inside the permutation
__global__ void kernel_create_mapping(const dist_t *distances, const int N,
                                      dist_t *regions, const int regions_count,
                                      dist_t *mapping) {

  int grid_size = blockDim.x * gridDim.x;
  int global_idx = blockDim.x * blockIdx.x + threadIdx.x;

  for (int i = global_idx; i < N; i += grid_size) {

    // Maps nodes without distance to the last region,
    // which is used only by them
    dist_t distance = distances[i];
    if (distance > regions_count)
      printf("AAAAAAAAAAAAAAAAA: %d\n", distance);

    int position = atomicAdd(&regions[distance], 1);
    mapping[i] = position;
  }
}

__global__ void kernel_permute_distances(const dist_t *read_distances,
                                         dist_t *write_distances,
                                         const dist_t *mapping, const int N) {

  int grid_size = blockDim.x * gridDim.x;
  int global_idx = blockDim.x * blockIdx.x + threadIdx.x;

  for (int i = global_idx; i < N; i += grid_size) {
    int mapped_i = mapping[i];
    write_distances[mapped_i] = read_distances[i];
  }
}

__global__ void kernel_make_permutation_map(const vert_t *permutation,
                                            vert_t *map, const int N) {

  int grid_size = blockDim.x * gridDim.x;
  int global_idx = blockDim.x * blockIdx.x + threadIdx.x;

  for (int i = global_idx; i < N; i += grid_size) {
    vert_t node_id = permutation[i];
    map[node_id] = i;
  }
}
*/

#define PERMUTATION_BLOCK_SIZE 1024
#define PERMUTATION_BLOCK_WORK 20
#define PERMUTATION_WORK (PERMUTATION_BLOCK_SIZE * PERMUTATION_BLOCK_WORK)

// Cool system... but a sort by key is faster... RIP
#define PERMUTATION_USE_HISTOGRAM 0
#define PERMUTATION_VALIDATE 1
#define PERMUTATION_VERBOSE 0

__global__ void dbfs_apply_relabel(const dist_t *read_dist, dist_t *write_dist,
                                   const dist_t *map, const int N) {
  int i;
  KERNEL_FOR_STRIDED(i, N) { write_dist[map[i]] = read_dist[i]; }
}

template <typename HornetGraph>
void DynamicBFS<HornetGraph>::relabel(
    const thrust::device_vector<vert_t> &map) {

  const vert_t *map_ptr = thrust::raw_pointer_cast(&map[0]);
  const int BC = BLOCK_COUNT(_graph.nV(), PERMUTATION_WORK);
  dbfs_apply_relabel<<<BC, PERMUTATION_BLOCK_SIZE>>>(
      current_distances(), write_distances(), map_ptr, _graph.nV());

  swap_and_sync_distances();

  // Obtain new source id, if it is a bfs cache relabeling it should be zero
  thrust::host_vector<vert_t> h_map = map;
  _source = h_map[_source];
  printf("dbfs new source after relabel is %d\n", _source);
}

/*
template <typename HornetGraph>
void DynamicBFS<HornetGraph>::apply_cache_reordering(bool sort_edges) {
#if PERMUTATION_USE_HISTOGRAM
  const int graph_blocks_count =
      (_graph.nV() + PERMUTATION_BLOCK_ITEMS - 1) / PERMUTATION_BLOCK_ITEMS;
  const int acc_blocks_count =
      (PARTIAL_HIST_SIZE + PERMUTATION_BLOCK_ITEMS - 1) /
      PERMUTATION_BLOCK_ITEMS;

  const int histogram_size = _current_level;
  thrust::device_vector<dist_t> histogram(histogram_size);
  thrust::device_vector<dist_t> partial_histograms(PARTIAL_HIST_SIZE *
                                                   graph_blocks_count);

#if PERMUTATION_VERBOSE
  printf("Launching permutation\n");
  printf("\thistogram_size: %d\n", histogram_size);
  printf("\tpartial_histograms_size: %dx%d\n", PARTIAL_HIST_SIZE,
         graph_blocks_count);
#endif

  // Divide total histogram in sections to better utilize shared memory
  for (int min_dist = 0; min_dist < histogram_size;
       min_dist += PARTIAL_HIST_SIZE) {

#if PERMUTATION_VERBOSE
    printf("Launching kernel for blocks histograms\n");
    printf("\tblocks: %d\n", graph_blocks_count);
    printf("\tmin:    %d\n", min_dist);
    printf("\tmax:    %d\n", min_dist + PARTIAL_HIST_SIZE);
#endif

    cudaDeviceSynchronize();
    dist_t *d_partial_histograms_ptr =
        thrust::raw_pointer_cast(&partial_histograms[0]);
    kernel_block_histogram<<<graph_blocks_count, PERMUTATION_BLOCK_SIZE>>>(
        current_distances(), _graph.nV(), d_partial_histograms_ptr, min_dist,
        min_dist + PARTIAL_HIST_SIZE);

#if PERMUTATION_VERBOSE
    printf("Launching kernel for histogram accumulation\n");
    printf("\tblocks: %d\n", acc_blocks_count);
#endif

    cudaDeviceSynchronize();
    dist_t *d_histogram_ptr = thrust::raw_pointer_cast(&histogram[min_dist]);
    kernel_accumulate_histogram<<<acc_blocks_count, PERMUTATION_BLOCK_SIZE>>>(
        d_partial_histograms_ptr, graph_blocks_count, d_histogram_ptr);
  }

#if PERMUTATION_VERBOSE
  printf("Calculated histogram\n");
  thrust::copy(histogram.begin(), histogram.end(),
               std::ostream_iterator<dist_t>(std::cout, " \n"));
#endif

  // Calculate the position of all distances nodes
  cudaDeviceSynchronize();
  thrust::exclusive_scan(thrust::device, histogram.begin(), histogram.end(),
                         histogram.begin());

#if PERMUTATION_VERBOSE
  printf("Calculated regions\n");
  thrust::copy(histogram.begin(), histogram.end(),
               std::ostream_iterator<dist_t>(std::cout, " \n"));
#endif

#if PERMUTATION_VERBOSE
  printf("Launching kernel for permutation creation\n");
  printf("\tblocks: %d\n", graph_blocks_count);
#endif

  cudaDeviceSynchronize();
  dist_t *d_histogram_ptr = thrust::raw_pointer_cast(&histogram[0]);
  kernel_create_mapping<<<graph_blocks_count, PERMUTATION_BLOCK_SIZE>>>(
      current_distances(), _graph.nV(), d_histogram_ptr, histogram_size,
      _relabeling); // This is slow!

  // Apply permutation to dynamic distance vector
  kernel_permute_distances<<<graph_blocks_count, PERMUTATION_BLOCK_SIZE>>>(
      current_distances(), write_distances(), _relabeling, _graph.nV());
  swap_distances();

#else

  const int nV = _graph.nV();

  thrust::device_vector<vert_t> sequence(nV);
  thrust::sequence(thrust::device, sequence.begin(), sequence.end(), 0);
  vert_t *sequence_ptr = thrust::raw_pointer_cast(&sequence[0]);

  thrust::device_vector<vert_t> permutation(nV);
  vert_t *permutation_ptr = thrust::raw_pointer_cast(&permutation[0]);

  size_t temp_buffer_size;
  cub::DeviceRadixSort::SortPairs(NULL, temp_buffer_size, current_distances(),
                                  write_distances(), sequence_ptr,
                                  permutation_ptr, nV);

  // Calculate where each node should go based on their distance
  // and update dynamic distance vector to the new ordering
  rmm::device_buffer temp_buffer{temp_buffer_size, rmm::cuda_stream_view{}};
  cub::DeviceRadixSort::SortPairs(temp_buffer.data(), temp_buffer_size,
                                  current_distances(), write_distances(),
                                  sequence_ptr, permutation_ptr, nV);
  swap_distances();

  // Calculate mapping of nodes given the permutation
  const int graph_blocks_count =
      (_graph.nV() + PERMUTATION_WORK - 1) / PERMUTATION_WORK;
  kernel_make_permutation_map<<<graph_blocks_count, PERMUTATION_BLOCK_SIZE>>>(
      permutation_ptr, _relabeling, nV);

#if PERMUTATION_VERBOSE
  thrust::host_vector<vert_t> h_perm = permutation;
  printf("Permutation:\n");
  for (int i = 0; i < nV; i++)
    printf("%2d ", h_perm[i]);

  printf("\nMapping:\n");
  for (int i = 0; i < nV; i++)
    printf("%2d ", i);
  printf("\n");

  thrust::host_vector<dist_t> relabeling(nV);
  thrust::device_ptr<dist_t> relabeling_ptr =
      thrust::device_pointer_cast(_relabeling);
  thrust::copy(relabeling_ptr, relabeling_ptr + nV, relabeling.begin());

  for (int i = 0; i < nV; i++)
    printf("%2d ", relabeling[i]);
  printf("\n");
#endif

#endif

  // Apply relabeling to entire graph
  _graph.relabel(_relabeling, sort_edges);

  // Relabel source node
  cudaMemcpy(&_source, _relabeling + _source, sizeof(dist_t),
             cudaMemcpyDeviceToHost);

#if PERMUTATION_VALIDATE
  int limit = 0;
  thrust::host_vector<dist_t> distances(_graph.nV());
  thrust::device_ptr<dist_t> ptr =
      thrust::device_pointer_cast(current_distances());
  thrust::copy(ptr, ptr + distances.size(), distances.begin());

#if PERMUTATION_VERBOSE
  printf("Distances:\n");
#endif
  for (int i = 0; i < _graph.nV(); i++) {
#if PERMUTATION_VERBOSE
    printf("%2d ", distances[i]);
#endif
    if (distances[i] > limit)
      limit = distances[i];
    else if (distances[i] < limit) {
      printf("PERMUTATION NOT CORRECT! limit: %d\n", limit);
      break;
    }
  }
#if PERMUTATION_VERBOSE
  printf("\n");
#endif
#endif
}
*/

#define DBFS_VERTEX_UPDATE_NO_ATOMIC 0
#if DBFS_VERTEX_UPDATE_NO_ATOMIC

// Calculate assigned new level to each destination
__global__ void kernel_edges_distances(vert_t *batch_src, dist_t *levels,
                                       dist_t *dist, int size) {

  const int stride = blockDim.x * gridDim.x;
  const int beg = blockDim.x * blockIdx.x + threadIdx.x;

  for (int i = beg; i < size; i += stride) {
    dist_t distance = dist[batch_src[i]];
    if (distance != INF)
      distance += 1;

    levels[i] = distance;

    /*
    printf("Edge distance[%3d] | src: %4d, assign_distance: %4d\n", i,
           batch_src[i], levels[i]);
    */
  }
}

// Apply delta to each destination vertex
__global__ void kernel_apply_distances(vert_t *unique_destinations,
                                       dist_t *unique_distances,
                                       dist_t *distances,
                                       TwoLevelQueue<vert_t> frontier,
                                       int size) {

  const int stride = blockDim.x * gridDim.x;
  const int beg = blockDim.x * blockIdx.x + threadIdx.x;

  for (int i = beg; i < size; i += stride) {
    vert_t dest_id = unique_destinations[i];

    /*
    printf("dest_id: %d, distance: %d, unique_distance: %d\n", dest_id,
           distances[dest_id], unique_distances[i]);
    */

    if (distances[dest_id] > unique_distances[i]) {
      distances[dest_id] = unique_distances[i];
      frontier.insert(dest_id);
    }
  }
}

#endif

template <typename HornetGraph>
void DynamicBFS<HornetGraph>::update(BatchUpdate &batch) {

  const int BATCH_SIZE = batch.size();
  _stats = {0};
  _device_timer.start();

#if DBFS_VERTEX_UPDATE_NO_ATOMIC
  auto soa = batch.in_edge().get_soa_ptr();
  vert_t *batch_src = soa.template get<0>();
  vert_t *batch_dst = soa.template get<1>();

  // Calculate new level of each edge destination in the batch
  thrust::device_vector<dist_t> distances(BATCH_SIZE);
  dist_t *distances_ptr = thrust::raw_pointer_cast(&distances[0]);
  kernel_edges_distances<<<xlib::ceil_div<1024>(BATCH_SIZE), 1024>>>(
      batch_src, distances_ptr, current_distances(), BATCH_SIZE);

  // Sort the distances based on their edge's destination, to compact them
  thrust::device_ptr<vert_t> dst_ptr = thrust::device_pointer_cast(batch_dst);
  thrust::device_vector<vert_t> destinations(dst_ptr, dst_ptr + BATCH_SIZE);
  thrust::sort_by_key(destinations.begin(), destinations.end(),
                      distances.begin());

  thrust::device_vector<vert_t> unique_destinations(BATCH_SIZE);
  thrust::device_vector<dist_t> unique_distances(BATCH_SIZE);

  // Calculate minimum new distance of all destination nodes
  auto new_end = thrust::reduce_by_key(
      thrust::device, destinations.begin(), destinations.end(),
      distances.begin(), unique_destinations.begin(), unique_distances.begin(),
      thrust::equal_to<vert_t>{}, thrust::minimum<int>{});

  vert_t *unique_destinations_ptr =
      thrust::raw_pointer_cast(&unique_destinations[0]);
  dist_t *unique_distances_ptr = thrust::raw_pointer_cast(&unique_distances[0]);

  const int effective_size = new_end.first - unique_destinations.begin();
  printf("Applying new distances for %d/%d of the batch edges\n",
         effective_size, BATCH_SIZE);

  // Apply new distances to nodes if possible
  sync_distances();
  kernel_apply_distances<<<xlib::ceil_div<1024>(effective_size), 1024>>>(
      unique_destinations_ptr, unique_distances_ptr, write_distances(),
      _frontier, effective_size);
#else
  // This is much faster than the other method
  sync_distances();
  forAllEdgesBatch(
      _graph, batch,
      BatchVertexUpdate{current_distances(), write_distances(), _frontier});
#endif

  swap_and_sync_distances();
  _frontier.swap();

  _device_timer.stop();
  _stats.vertex_update_time = _device_timer.duration();

  std::vector<int> frontier_sizes;
  frontier_sizes.reserve(1024);

  _device_timer.start();

  _stats.initial_frontier_size = _frontier.size();
  while (_frontier.size() != 0) {

    frontier_sizes.push_back(_frontier.size());
    _stats.frontier_expansions_count += 1;

    // Propagate changes to all children
    forAllEdges(
        _graph, _frontier,
        DynamicExpandEdge{current_distances(), write_distances(), _frontier},
        _load_balancing);

    swap_and_sync_distances();
    _frontier.swap();
  }

  _device_timer.stop();

  _stats.expansion_time = _device_timer.duration();
  _stats.min_frontier_size =
      *std::min_element(frontier_sizes.begin(), frontier_sizes.end());
  _stats.max_frontier_size =
      *std::max_element(frontier_sizes.begin(), frontier_sizes.end());

  // Calculate median frontier size
  _stats.median_frontier_size = 0;
  if (!frontier_sizes.empty()) {

    const auto middle = frontier_sizes.begin() + frontier_sizes.size() / 2;
    std::nth_element(frontier_sizes.begin(), middle, frontier_sizes.end());

    _stats.median_frontier_size = *middle;
    if (frontier_sizes.size() % 2 == 0) {
      const auto left_middle = std::max_element(frontier_sizes.begin(), middle);
      _stats.median_frontier_size = (*left_middle + *middle) / 2;
    }
  }
}

struct EraseBatchUpdate {

  int *flagged;
  dist_t *read_distances;
  TwoLevelQueue<vert_t> origins;

  OPERATOR(Vertex &src, Vertex &dst) {

    const int src_distance = read_distances[src.id()];
    const int dst_distance = read_distances[dst.id()];

    // printf("Check edge: %d(%d) -> %d(%d)\n", src.id(), src_distance,
    // dst.id(),
    //        dst_distance);

    if (dst_distance > src_distance) {
      // printf("Found origin: %d\n", dst.id());
      origins.insert(dst.id());
      atomicExch(flagged + dst.id(), 1);
    }
  }
};

struct FindPrecursors {
  int *flagged;
  dist_t *read_distances;
  TwoLevelQueue<vert_t> frontier;
  TwoLevelQueue<vert_t> precursors;

  OPERATOR(Vertex &src, Edge &edge) {

    const int src_distance = read_distances[edge.src_id()];
    const int dst_distance = read_distances[edge.dst_id()];

    // printf("Check edge: %d(%d) -> %d(%d)\n", edge.src_id(), src_distance,
    //        edge.dst_id(), dst_distance);

    vert_t dst_id = edge.dst_id();
    if (dst_distance > src_distance) {
      // We found a node under the influence of an origin, we must continue
      // the search to a precursors
      // printf("continue search from %d\n", dst_id);
      frontier.insert(dst_id);
      atomicExch(flagged + dst_id, 1);
    } else {
      // The destination node is from another origin, so it's a precursor
      // printf("found precursor: %d\n", dst_id);
      precursors.insert(dst_id);
    }
  }
};

struct FilterPrecursors {

  int *flagged;
  TwoLevelQueue<vert_t> frontier;

  OPERATOR(vert_t id) {
    if (flagged[id] == 0)
      frontier.insert(id);
  }
};

struct ResetDistance {
  int *flagged;
  dist_t *write_distances;
  OPERATOR(Vertex &vertex) {
    vert_t id = vertex.id();
    if (flagged[id] != 0 && id != 512 && id != 1280) {
      // printf("flagged: %d\n", id);
      write_distances[id] = INF;
    }
  }
};

struct PropagatePrecursors {

  dist_t *read_distances, *write_distances;
  TwoLevelQueue<vert_t> frontier;

  OPERATOR(Vertex &src, Edge &edge) {

    const int src_dist = read_distances[edge.src_id()];
    const int dst_dist = read_distances[edge.dst_id()];

    vert_t dst_id = edge.dst_id();
    if (dst_dist > src_dist + 1) {

      // Modify distance of destination node, if the operation returns our read
      // value then we must add to queue
      if (atomicMin(&write_distances[dst_id], src_dist + 1) == dst_dist) {
        frontier.insert(dst_id);
      }
    }
  }
};

template <typename HornetGraph>
void DynamicBFS<HornetGraph>::erase(BatchUpdate &batch, bool verbose) {

  const int BATCH_SIZE = batch.size();
  sync_distances();

  // 1) Add to the frontier only the nodes that had one of their useful edges
  // removed
  // 2) Propagate from the remaining nodes to their precursors adding
  // them to a queue and flagging all traversed nodes as modifiable.
  // 3) Eliminate all precursors that are modifiable nodes
  // 4) Propagate distances as normal from precursors

  thrust::device_vector<int> flagged(_graph.nV(), 0);
  int *flagged_ptr = thrust::raw_pointer_cast(&flagged[0]);

  _frontier.clear();
  forAllEdgesBatch(
      _graph, batch,
      EraseBatchUpdate{flagged_ptr, current_distances(), _frontier});

  _frontier.swap();
  if (verbose) {
    printf("Origin nodes count: %d\n", _frontier.size());
    _frontier.print();
  }

  // Find all precursors
  TwoLevelQueue<vert_t> precursors{_graph, 10.0f};
  while (_frontier.size() != 0) {
    forAllEdges(
        _graph, _frontier,
        FindPrecursors{flagged_ptr, current_distances(), _frontier, precursors},
        _load_balancing);
    _frontier.swap();
  }

  precursors.swap();
  if (verbose) {
    printf("Found this precursors(unfiltered): %d\n", precursors.size());
    precursors.print();
  }

  // Remove precursors that can't be used because they are under the influence
  // of an origin, that is, they are flagged
  forAll(precursors, FilterPrecursors{flagged_ptr, _frontier});

  _frontier.swap();
  if (verbose) {
    printf("Found this precursors(filtered): %d\n", _frontier.size());
    _frontier.print();
  }

  // Reset the distance of all flagged nodes to INF
  forAllVertices(_graph, ResetDistance{flagged_ptr, write_distances()});
  swap_and_sync_distances();

  // If there are no precursors then we have finished
  if (precursors.size() == 0) {
    if (verbose)
      printf("No precursors, stopping\n");
    return;
  }

  // Propagate from the precursors normally
  while (_frontier.size() != 0) {
    if (verbose)
      printf("Expanding with %d elements\n", _frontier.size());

    forAllEdges(
        _graph, _frontier,
        PropagatePrecursors{current_distances(), write_distances(), _frontier},
        _load_balancing);

    swap_and_sync_distances();
    _frontier.swap();
  }
}

template <typename HornetGraph>
void DynamicBFS<HornetGraph>::update_bottom_up(BatchUpdate &batch) {
  const int BATCH_SIZE = batch.size();
  _device_timer.start();

  int *work_to_do;
  cudaMalloc(&work_to_do, sizeof(int));
  int host_work_to_do;

  sync_distances();
  forAllEdgesBatch(_graph, batch,
                   BUBatchVertexUpdate{current_distances(), write_distances()});
  swap_and_sync_distances();

  _device_timer.stop();
  _stats.vertex_update_time = _device_timer.duration();
  _device_timer.start();

  do {
    cudaMemset(work_to_do, 0, sizeof(char));
    forAllEdges(
        _parent_graph,
        BUDynamicExpandEdge{work_to_do, current_distances(), write_distances()},
        _load_balancing);
    swap_and_sync_distances();

    // Check if there is more work to do
    cudaMemcpy(&host_work_to_do, work_to_do, sizeof(char),
               cudaMemcpyDeviceToHost);

  } while (host_work_to_do != 0);

  _device_timer.stop();
  _stats.expansion_time = _device_timer.duration();
  cudaFree(&work_to_do);
}

#if 0
#define CSV_DBFS_OUTPUT
#endif

/*
template <typename HornetGraph>
void DynamicBFS<HornetGraph>::update(const vert_t *dst_vertices, int count) {
  sync_distances();

  // Used to benchmark all code parts
  timer::Timer<timer::DEVICE> section_timer;

#ifdef CSV_DBFS_OUTPUT
  std::cerr << "seq\tfrontier_size" << std::endl;
  int index = 0;
#endif

  section_timer.start();

  // Find smallest parents
  _frontier.insert(dst_vertices, count);
  forAllEdges(_parent_graph, _frontier,
              VertexUpdate{current_distances(), write_distances(), _frontier},
              _load_balancing);

  swap_and_sync_distances();
  _frontier.swap();

  section_timer.stop();

  _stats.vertex_update_time = section_timer.duration();
  // section_timer.print("Vertex Update");

  section_timer.start();

  // Propagate change to all nodes
  _stats.initial_frontier_size = _frontier.size();
  while (_frontier.size() > 0) {
    // cudaDeviceSynchronize();

    const int frontier_size = _frontier.size();
    _stats.frontier_expansions_count += 1;

#ifdef CSV_DBFS_OUTPUT
    std::cerr << index << "\t" << frontier_size << std::endl;
    index += 1;
#endif

    // Propagate new distances
    forAllEdges(
        _graph, _frontier,
        DynamicExpandEdge{current_distances(), write_distances(), _frontier},
        _load_balancing);

    swap_and_sync_distances();
    _frontier.swap();
  }

  section_timer.stop();

  _stats.expansion_time = section_timer.duration();
  // section_timer.print("Dynamic expand");
}
*/

template <typename HornetGraph> void DynamicBFS<HornetGraph>::release() {
  gpu::free(_distances[0], _graph.nV());
  gpu::free(_distances[1], _graph.nV());
  gpu::free(_relabeling, _graph.nV());
}

template <typename HornetGraph> bool DynamicBFS<HornetGraph>::validate() {
  const auto nV = StaticAlgorithm<HornetGraph>::hornet.nV();

  // Launch static BFS on the current graph
  BfsTopDown2<HornetGraph> BFS(_graph);
  BFS.set_parameters(_source);

  _device_timer.start();
  BFS.run();
  _device_timer.stop();

  _stats.bfs_stats = BFS.get_stats();
  _stats.bfs_time = _device_timer.duration();
  _stats.bfs_max_level = BFS.getLevels();

#if 0
  thrust::device_ptr<dist_t> bfs_dist =
      thrust::device_pointer_cast(BFS.get_distance_vector());

  thrust::device_ptr<dist_t> dbfs_dist =
      thrust::device_pointer_cast(current_distances());

  bool equal =
      thrust::equal(thrust::device, bfs_dist, bfs_dist + nV, dbfs_dist);
  return equal;
#endif

#if 1

  // Copy normal BFS distance vector
  auto *nbfs_host_distances = new dist_t[nV];
  gpu::copyToHost(BFS.get_distance_vector(), nV, nbfs_host_distances);

#if 0
  for (int i = 0; i < nV; i++)
    printf("[NBFS] node: %d | dist: %d\n", i, nbfs_host_distances[i]);
#endif

  // Copy dynamic BFS distance vector
  auto *dbfs_host_distances = new dist_t[nV];
  gpu::copyToHost(current_distances(), nV, dbfs_host_distances);

#if 0
  for (int i = 0; i < nV; i++)
    printf("[DBFS] node: %d | dist: %d\n", i, dbfs_host_distances[i]);
#endif

  // They must be the same
  int error_count = 0;
  for (int i = 0; i < nV; i++) {

    if (nbfs_host_distances[i] != dbfs_host_distances[i]) {
      error_count += 1;
      printf("[%6d] vertex %6d | nbfs: %6d, dbfs: %6d\n", error_count, i,
             nbfs_host_distances[i], dbfs_host_distances[i]);
#if 0
      if (error_count >= 10) {
        printf("... and other errors\n");
        break;
      }
#endif
    }
  }

  delete[] nbfs_host_distances;
  delete[] dbfs_host_distances;

  return error_count == 0;
#endif
};

} // namespace hornets_nest
