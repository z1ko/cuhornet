
#include <BufferPool.cuh>
#include <Device/Util/Timer.cuh>
#include <HornetAlg.hpp>
#include <Static/BreadthFirstSearch/TopDown2.cuh>
#include <cstdio>
#include <cuda_device_runtime_api.h>
#include <iterator>
#include <stdio.h>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

#include <Macros.hpp>

namespace hornets_nest {
// clang-format off

using vert_t = int;
using dist_t = int;

/**
 * @brief Relabel the graph vertices in order to obtain
 * a more cache friendly ordering
 */
template <typename Graph> class BFSRelabel {

  using BatchUpdate = hornet::gpu::BatchUpdate<vert_t>;
  using BatchUpdatePtrDevice =
    hornet::BatchUpdatePtr<vert_t, hornet::EMPTY, hornet::DeviceType::DEVICE>;

public:
  struct Stats {
    float sorting_time;
    float generation_time;
    float applying_time;
  };

  /// @brief Apply a random mapping to the graph
  /// @return The random mapping
  thrust::device_vector<vert_t> randomize(Graph& graph);

  /// @brief Apply an ordering based on BFS visit order.
  /// @param source The source used in the BFS
  /// @param sort_edges Also sort all adj list.
  /// @return The applied mapping 
  thrust::device_vector<vert_t> relabel(Graph& graph, vert_t source, bool sort_edges = false);

  /// @brief Apply ordering to a vector
  void apply(const thrust::device_vector<vert_t>& map, vert_t* d_target, vert_t* d_output, int N);

  /// @brief Apply ordering to batch update
  BatchUpdate apply(BatchUpdate& batch, const thrust::device_vector<vert_t>& map);

  /// @brief Apply ordering to graph
  void apply(Graph& graph, thrust::device_vector<vert_t>& map);

  /// @brief obtain inverse map
  thrust::device_vector<vert_t> inverse(const thrust::device_vector<vert_t>& map);

  inline Stats get_stats() { return _stats; }

private:
  timer::Timer<timer::DEVICE> _timer;
  Stats _stats;
};

#define BFS_RELABEL_MAPPING_BLOCK_SIZE 1024
#define BFS_RELABEL_MAPPING_BLOCK_WORK 20
#define BFS_RELABEL_MAPPING_WORK \
  (BFS_RELABEL_MAPPING_BLOCK_SIZE * BFS_RELABEL_MAPPING_BLOCK_WORK)

__global__ void bfs_relabel_map(const vert_t* map,
                                vert_t* result, const int N) {
  
  int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  int grid_size = blockDim.x * gridDim.x;

  for(int i = global_idx; i < N; i += grid_size) {
    vert_t node_id = map[i];
    result[node_id] = i;
  }
}

template<typename Graph>
thrust::device_vector<vert_t> BFSRelabel<Graph>::randomize(Graph& graph) {

  _timer.start();

  thrust::device_vector<vert_t> mapping(graph.nV());
  vert_t* mapping_ptr = thrust::raw_pointer_cast(&mapping[0]);
  thrust::sequence(thrust::device, mapping.begin(), 
                   mapping.end(), 0);

  static thrust::default_random_engine rng;
  thrust::device_vector<vert_t> permutation(graph.nV());
  vert_t* permutation_ptr = thrust::raw_pointer_cast(&permutation[0]);
  thrust::shuffle_copy(thrust::device, mapping.begin(), mapping.end(), 
                       permutation.begin(), rng);

  // Print permutation
#if 0
  printf("Graph permutation:\n\n");
  thrust::host_vector<vert_t> hp = permutation;
  for (vert_t id : hp) printf("%2d ", id);
  printf("\n=========================\n");
#endif

  // Create mapping from permutation
  const int BC = BLOCK_COUNT(graph.nV(), BFS_RELABEL_MAPPING_WORK);
  bfs_relabel_map<<<BC, BFS_RELABEL_MAPPING_BLOCK_SIZE>>>(
    permutation_ptr, mapping_ptr, graph.nV());

  _timer.stop();
  _stats.generation_time = _timer.duration();

  INSTRUMENT(_timer, _stats.applying_time, 
    graph.relabel(mapping_ptr, false));

  return mapping;
}

template<typename Graph>
thrust::device_vector<vert_t> BFSRelabel<Graph>::relabel(Graph& graph, vert_t source, bool sort_edges) {

  BfsTopDown2<Graph> bfs{graph};
  bfs.set_parameters(source);
  bfs.run();


  thrust::device_vector<vert_t> permutation(graph.nV());
  thrust::device_vector<dist_t> ordered_distances(graph.nV());
  thrust::device_vector<vert_t> mapping(graph.nV());
  thrust::sequence(thrust::device, mapping.begin(), 
                   mapping.end(), 0);

  vert_t *permutation_ptr = thrust::raw_pointer_cast(&permutation[0]);
  dist_t* ordered_distances_ptr = thrust::raw_pointer_cast(&ordered_distances[0]);
  vert_t* mapping_ptr = thrust::raw_pointer_cast(&mapping[0]);
  
  _timer.start();
  size_t required_memory;
  cub::DeviceRadixSort::SortPairs(NULL, required_memory, bfs.get_distance_vector(), 
                                  ordered_distances_ptr, mapping_ptr, permutation_ptr, graph.nV());
  rmm::device_buffer memory{required_memory, rmm::cuda_stream_view{}};

  // Calculate where each node should go based on their distance
  cub::DeviceRadixSort::SortPairs(memory.data(), required_memory,
                                  bfs.get_distance_vector(), ordered_distances_ptr,
                                  mapping_ptr, permutation_ptr, graph.nV());
 
  _timer.stop();
  _stats.sorting_time = _timer.duration();
  _timer.start();

  // Create mapping from permutation
  const int BC = BLOCK_COUNT(graph.nV(), BFS_RELABEL_MAPPING_WORK);
  bfs_relabel_map<<<BC, BFS_RELABEL_MAPPING_BLOCK_SIZE>>>(
    permutation_ptr, mapping_ptr, graph.nV());

  _timer.stop();
  _stats.generation_time = _timer.duration();

  // Check that the ordering is correct
#if 1
  dist_t bound = 0;
  thrust::host_vector<vert_t> distances = ordered_distances;
  for (vert_t distance : distances) {
    if (distance >= bound) bound = distance;
    else {
      printf("ERROR: Order generated by relabel is not correct!\n");
      exit(1);
    }
  }
#endif

  INSTRUMENT(_timer, _stats.applying_time, 
    graph.relabel(mapping_ptr, sort_edges));

  return mapping;
}

__global__ void bfs_relabel_apply(const vert_t* map, vert_t* target, 
                                  vert_t* output, const int N) {
  int i;
  KERNEL_FOR_STRIDED(i, N) {
    output[i] = map[target[i]];
  }
}

template<typename Graph>
void BFSRelabel<Graph>::apply(const thrust::device_vector<vert_t>& map, 
                              vert_t* d_target, vert_t* d_output, int N) {

  const vert_t* map_ptr = thrust::raw_pointer_cast(&map[0]);
  const int BC = BLOCK_COUNT(N, BFS_RELABEL_MAPPING_WORK);
  bfs_relabel_apply<<<BC, BFS_RELABEL_MAPPING_BLOCK_SIZE>>>(
    map_ptr, d_target, d_output, N);
}

template<typename Graph>
void BFSRelabel<Graph>::apply(Graph& graph, thrust::device_vector<vert_t>& map) {

  vert_t* map_ptr = thrust::raw_pointer_cast(&map[0]);
  const int BC = BLOCK_COUNT(graph.nV(), BFS_RELABEL_MAPPING_WORK);
  graph.relabel(map_ptr, false);
}

template<typename Graph>
BatchUpdate BFSRelabel<Graph>::apply(BatchUpdate& batch, const thrust::device_vector<vert_t>& map) {
  
  auto soa = batch.in_edge().get_soa_ptr();
  thrust::device_vector<vert_t> src(batch.size());
  vert_t *src_ptr = thrust::raw_pointer_cast(&src[0]);
  thrust::device_vector<vert_t> dst(batch.size());
  vert_t *dst_ptr = thrust::raw_pointer_cast(&dst[0]);

  apply(map, soa.template get<0>(), src_ptr, batch.size());
  apply(map, soa.template get<1>(), dst_ptr, batch.size());

  BatchUpdatePtrDevice data{batch.size(), src_ptr, dst_ptr};
  return BatchUpdate{data};
}

template<typename Graph>
thrust::device_vector<vert_t> BFSRelabel<Graph>::inverse(const thrust::device_vector<vert_t>& map) {
  
  thrust::device_vector<vert_t> result(map.size());
  vert_t* result_ptr = thrust::raw_pointer_cast(&result[0]);

  const vert_t* map_ptr = thrust::raw_pointer_cast(&map[0]);
  const int BC = BLOCK_COUNT(map.size(), BFS_RELABEL_MAPPING_WORK);
  bfs_relabel_map<<<BC, BFS_RELABEL_MAPPING_BLOCK_SIZE>>>(
    map_ptr, result_ptr, map.size());

  return result;
}

// clang-format on
} // namespace hornets_nest
