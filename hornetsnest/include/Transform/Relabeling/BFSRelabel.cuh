
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
public:
  struct Stats {
    float generation_time;
    float applying_time;
  };

public:
  BFSRelabel(Graph &graph);

  /// @brief Apply a random mapping to the graph
  /// @return The random mapping
  thrust::device_vector<vert_t> randomize();

  /// @brief Apply an ordering based on BFS visit order.
  /// @param source The source used in the BFS
  /// @param sort_edges Also sort all adj list.
  /// @return The applied mapping 
  thrust::device_vector<vert_t> relabel(vert_t source, bool sort_edges = false);

  inline Stats get_stats() { return _stats; }

private:
  timer::Timer<timer::DEVICE> _timer;
  Stats _stats;

  // Base algorithm used to obtain a
  // ordering of the nodes base on distance
  BfsTopDown2<Graph> _bfs;
  float _relabel_time;

  Graph &_graph;
};

template <typename Graph>
BFSRelabel<Graph>::BFSRelabel(Graph &graph) 
  : _graph{graph}, _bfs{graph} {}

#define BFS_RELABEL_MAPPING_BLOCK_SIZE 1024
#define BFS_RELABEL_MAPPING_BLOCK_WORK 20
#define BFS_RELABEL_MAPPING_WORK \
  (BFS_RELABEL_MAPPING_BLOCK_SIZE * BFS_RELABEL_MAPPING_BLOCK_WORK)

__global__ void bfs_relabel_map(const vert_t* permutation,
                                vert_t* result, const int N) {
  
  int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  int grid_size = blockDim.x * gridDim.x;

  for(int i = global_idx; i < N; i += grid_size) {
    vert_t node_id = permutation[i];
    result[node_id] = i;
  }
}

template<typename Graph>
thrust::device_vector<vert_t> BFSRelabel<Graph>::randomize() {

  _timer.start();

  thrust::device_vector<vert_t> mapping(_graph.nV());
  vert_t* mapping_ptr = thrust::raw_pointer_cast(&mapping[0]);
  thrust::sequence(thrust::device, mapping.begin(), 
                   mapping.end(), 0);

  thrust::default_random_engine rng;
  thrust::device_vector<vert_t> permutation(_graph.nV());
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
  const int BC = BLOCK_COUNT(_graph.nV(), BFS_RELABEL_MAPPING_WORK);
  bfs_relabel_map<<<BC, BFS_RELABEL_MAPPING_BLOCK_SIZE>>>(
    permutation_ptr, mapping_ptr, _graph.nV());

  _timer.stop();
  _stats.generation_time = _timer.duration();

  INSTRUMENT(_timer, _stats.applying_time, 
    _graph.relabel(mapping_ptr, false));

  return mapping;
}

template<typename Graph>
thrust::device_vector<vert_t> BFSRelabel<Graph>::relabel(vert_t source, bool sort_edges) {

  _bfs.reset();
  _bfs.set_parameters(source);
  _bfs.run();

  _timer.start();

  thrust::device_vector<vert_t> permutation(_graph.nV());
  thrust::device_vector<dist_t> ordered_distances(_graph.nV());
  thrust::device_vector<vert_t> mapping(_graph.nV());
  thrust::sequence(thrust::device, mapping.begin(), 
                   mapping.end(), 0);

  vert_t *permutation_ptr = thrust::raw_pointer_cast(&permutation[0]);
  dist_t* ordered_distances_ptr = thrust::raw_pointer_cast(&ordered_distances[0]);
  vert_t* mapping_ptr = thrust::raw_pointer_cast(&mapping[0]);
  
  size_t required_memory;
  cub::DeviceRadixSort::SortPairs(NULL, required_memory, _bfs.get_distance_vector(), 
                                  ordered_distances_ptr, mapping_ptr, permutation_ptr, _graph.nV());
  rmm::device_buffer memory{required_memory, rmm::cuda_stream_view{}};

  // Calculate where each node should go based on their distance
  cub::DeviceRadixSort::SortPairs(memory.data(), required_memory,
                                  _bfs.get_distance_vector(), ordered_distances_ptr,
                                  mapping_ptr, permutation_ptr, _graph.nV());
  
  // Create mapping from permutation
  const int BC = BLOCK_COUNT(_graph.nV(), BFS_RELABEL_MAPPING_WORK);
  bfs_relabel_map<<<BC, BFS_RELABEL_MAPPING_BLOCK_SIZE>>>(
    permutation_ptr, mapping_ptr, _graph.nV());

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
    _graph.relabel(mapping_ptr, sort_edges));

  return mapping;
}

// clang-format on
} // namespace hornets_nest
