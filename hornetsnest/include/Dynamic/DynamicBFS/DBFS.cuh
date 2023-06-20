#pragma once

#include "HornetAlg.hpp"
#include <BufferPool.cuh>
#include <Device/Util/Timer.cuh>
#include <LoadBalancing/LogarithmRadixBinning.cuh>
#include <Static/BreadthFirstSearch/TopDown2.cuh>
#include <iterator>
#include <stdio.h>

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

    // Time measures for dynamic update
    float vertex_update_time;
    float expansion_time;

    float bfs_time;
    int bfs_max_level;
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

  /// @brief Process the inserted or removed edges to generate an updated
  /// distance array
  void update(BatchUpdate &batch);

  /// @brief Use the distance vector to calculate a permutation
  /// of the nodes that is cache friendly and apply it to the graph.
  void apply_cache_reordering(bool apply_to_edges = false);

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
  dist_t *_permutation{nullptr};
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

//------------------------------------------------------------------------------

template <typename HornetGraph>
DynamicBFS<HornetGraph>::DynamicBFS(HornetGraph &graph,
                                    HornetGraph &parent_graph)
    : StaticAlgorithm<HornetGraph>{graph}, _load_balancing{graph},
      _frontier{graph, 10.0f}, _graph{graph}, _parent_graph{parent_graph} {
  _buffer_pool.allocate(&_distances[0], _graph.nV());
  _buffer_pool.allocate(&_distances[1], _graph.nV());
  _buffer_pool.allocate(&_permutation, _graph.nV());
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
__global__ void sync_distances_kernel(const dist_t *from, dist_t *to,
                                      int size) {

  int stride = blockDim.x * gridDim.x;
  int start = blockIdx.x * blockDim.x + threadIdx.x;

  for (auto i = start; i < size; i += stride) {
    to[i] = from[i];
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
__global__ void kernel_create_permutation(const dist_t *distances, const int N,
                                          dist_t *regions,
                                          const int regions_count,
                                          dist_t *permutation) {

  int grid_size = blockDim.x * gridDim.x;
  int global_idx = blockDim.x * blockIdx.x + threadIdx.x;

  for (int i = global_idx; i < N; i += grid_size) {
    dist_t distance = distances[global_idx];
    if (distance < regions_count) {
      int position = atomicAdd(&regions[distance], 1);
      permutation[position] = global_idx;
    }
  }
}

#define PERMUTATION_BLOCK_SIZE 1024
#define PERMUTATION_BLOCK_WORK 20
#define PERMUTATION_BLOCK_ITEMS                                                \
  (PERMUTATION_BLOCK_SIZE * PERMUTATION_BLOCK_WORK)

/// Cool system... but a sort by key is faster... RIP
#define PERMUTATION_USE_HISTOGRAM 1
#define PERMUTATION_VERBOSE 0

template <typename HornetGraph>
void DynamicBFS<HornetGraph>::apply_cache_reordering(bool apply_to_edges) {

  timer::Timer<timer::HOST> timer;
  timer.start();

#if PERMUTATION_USE_HISTOGRAM

  const int graph_blocks_count =
      (_graph.nV() + PERMUTATION_BLOCK_ITEMS - 1) / PERMUTATION_BLOCK_ITEMS;
  const int acc_blocks_count =
      (PARTIAL_HIST_SIZE + PERMUTATION_BLOCK_ITEMS - 1) /
      PERMUTATION_BLOCK_ITEMS;

  const int histogram_size = _current_level - 1;
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
  kernel_create_permutation<<<graph_blocks_count, PERMUTATION_BLOCK_SIZE>>>(
      current_distances(), _graph.nV(), d_histogram_ptr, histogram_size,
      _permutation); // This is slow!

#else

  sync_distances();
  thrust::sequence(thrust::device, _permutation, _permutation + _graph.nV());
  thrust::sort_by_key(thrust::device, write_distances(),
                      write_distances() + _graph.nV(), _permutation);
  swap_distances();

#endif

  // Apply permutation to hornet
  _graph.permute(_permutation);

  // Permute source node
  cudaMemcpy(&_source, _permutation + _source, sizeof(dist_t),
             cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  timer.stop();
  printf("Permutation time: %f\n", timer.duration());
}

#ifdef DBFS_FAST

// Calculate delta of each edge
__global__ void kernel_edges_delta(int *batch_src, int *batch_dst, int *delta,
                                   int *dist, int size) {

  const int stride = blockDim.x * gridDim.x;
  const int beg = blockDim.x * blockIdx.x + threadIdx.x;

  for (int i = beg; i < size; i += stride) {
    delta[i] = dist[batch_dst[i]] - dist[batch_src[i]];
  }
}

// Apply delta to each destination vertex
__global__ void kernel_apply_delta(int *vertices_ptr, int *delta_ptr, int *dist,
                                   int size) {

  const int stride = blockDim.x * gridDim.x;
  const int beg = blockDim.x * blockIdx.x + threadIdx.x;

  for (int i = beg; i < size; i += stride) {
    int effective_delta = max(delta_ptr[i] - 1, 0);
    dist[vertices_ptr[i]] -= effective_delta;
  }
}

#endif

template <typename HornetGraph>
void DynamicBFS<HornetGraph>::update(BatchUpdate &batch) {
  const int BATCH_SIZE = batch.size();

  _device_timer.start();
  sync_distances();

  // Process batch by filtering edges with delta <= 1
  // the remaining destination nodes are added to the frontier
#ifndef DBFS_FAST
  forAllEdgesBatch(
      _graph, batch,
      BatchVertexUpdate{current_distances(), write_distances(), _frontier});
#else

  auto batch_soa_ptr = update.in_edge().get_soa_ptr();
  vert_t *batch_src = batch_soa_ptr.template get<0>();
  vert_t *batch_dst = batch_soa_ptr.template get<1>();

  // Calculate delta of each edge in the batch
  thrust::device_vector<int> delta(BATCH_SIZE);
  int *delta_ptr = thrust::raw_pointer_cast(&delta[0]);
  kernel_edges_delta<<<xlib::ceil_div<1024>(BATCH_SIZE), 1024>>>(
      batch_src, batch_src, delta_ptr, _distances, batch.size())

      // Sort the deltas based on their edge's destination
      thrust::device_vector<vert_t>
          vertices(batch_dst, batch_dst + BATCH_SIZE);
  thrust::sort_by_key(vertices.begin(), vertices.end(), delta.begin());

  thrust::device_vector<vert_t> unique_vertices(BATCH_SIZE);
  thrust::device_vector<int> unique_delta(BATCH_SIZE);

  // Calculate maximum delta of all destination nodes
  thrust::pair<vert_t *, int *> new_end;
  new_end = thrust::reduce_by_key(
      thrust::device, vertices.begin(), vertices.end(), delta.begin(),
      unique_vertices.begin(), unique_delta.begin(), thrust::equal_to<vert_t>,
      thrust::maximum<int>);

  // Apply delta to destination vertices
  delta_ptr = thrust::raw_pointer_cast(&unique_delta[0]);
  vert_t *vertices_ptr = thrust::raw_pointer_cast(&vertices[0]);
  kernel_apply_delta<<<xlib::ceil_div<1024>(BATCH_SIZE), 1024>>>(
      vertices_ptr, delta_ptr, _distances, new_end.first);

#endif

  swap_and_sync_distances();
  _frontier.swap();

  _device_timer.stop();
  _stats.vertex_update_time = _device_timer.duration();
  _device_timer.start();

  _stats.initial_frontier_size = _frontier.size();
  while (_frontier.size() != 0) {
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
  gpu::free(_permutation, _graph.nV());
}

template <typename HornetGraph> bool DynamicBFS<HornetGraph>::validate() {
  const auto nV = StaticAlgorithm<HornetGraph>::hornet.nV();

  // Launch static BFS on the current graph
  BfsTopDown2<HornetGraph> BFS(_graph);
  BFS.set_parameters(_source);

  _device_timer.start();
  BFS.run();
  _device_timer.stop();

  _stats.bfs_time = _device_timer.duration();
  _stats.bfs_max_level = BFS.getLevels();

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

    // If the nbfs distance is INF then the node is not in reachable from source
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
};

} // namespace hornets_nest
