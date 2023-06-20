/**
 * @file DBFS.cu
 * @brief Test of Dynamic BFS
 */

#include <cmath>
#include <iostream>

#include <Graph/GraphStd.hpp>
#include <StandardAPI.hpp>
#include <Util/BatchFunctions.hpp>
#include <Util/CommandLineParam.hpp>

#include "Dynamic/DynamicBFS/DBFS.cuh"

namespace test {

using namespace hornets_nest;
using namespace graph::structure_prop;
using namespace graph::parsing_prop;

using HornetInit = ::hornet::HornetInit<vert_t>;
using HornetGraph = ::hornet::gpu::Hornet<vert_t>;
using HornetBatchUpdatePtr =
    hornet::BatchUpdatePtr<vert_t, hornet::EMPTY, hornet::DeviceType::DEVICE>;
using HornetBatchUpdate = hornet::gpu::BatchUpdate<vert_t>;

// Generate a batch that requires an expansion
void generateEvilBatch(vert_t *src, vert_t *dst, int batch_size, dist_t *dist,
                       int dist_size, int delta) {
  srand(time(0));
  for (int i = 0; i < batch_size; i++) {

    vert_t src_id = rand() % dist_size;
    vert_t dst_id = rand() % dist_size;

    /*
    while(dist[src_id] == INF || dist[dst_id] - dist[src_id] < delta) {
        src_id = rand() % dist_size;
        dst_id = rand() % dist_size;
    }
    */

    src[i] = src_id;
    dst[i] = dst_id;
  }
}

int exec(int argc, char **argv) {

  graph::GraphStd<vert_t, vert_t> host_graph(ENABLE_INGOING);
  host_graph.read(argv[1], PRINT_INFO);
  int batch_size = std::stoi(argv[2]);

  HornetInit graph_init{host_graph.nV(), host_graph.nE(),
                        host_graph.csr_out_offsets(),
                        host_graph.csr_out_edges()};

  HornetInit graph_init_inv{host_graph.nV(), host_graph.nE(),
                            host_graph.csr_in_offsets(),
                            host_graph.csr_in_edges()};

  HornetGraph device_graph{graph_init};
  HornetGraph device_graph_inv{graph_init_inv};
  DynamicBFS<HornetGraph> DBFS{device_graph, device_graph_inv};

  vert_t source = device_graph.max_degree_id();
  DBFS.set_source(0 /*source*/);
  DBFS.run();

  printf("Graph before reordering:\n");
  device_graph.print();

  // Try permutation
  DBFS.apply_cache_reordering();

  printf("Graph after reordering: \n");
  device_graph.print();

  return 0;

  // =======================================================================
  // Create and apply new batch undirected

  vert_t *batch_src = new vert_t[batch_size];
  vert_t *batch_dst = new vert_t[batch_size];

  vert_t *dev_batch_src, *dev_batch_dst;
  cudaMalloc(&dev_batch_src, sizeof(vert_t) * batch_size);
  cudaMalloc(&dev_batch_dst, sizeof(vert_t) * batch_size);

  printf("Generating batch of %d edges\n", batch_size);
#if 0
    generateBatch(host_graph, batch_size, batch_src, batch_dst, BatchGenType::INSERT, batch_gen_property::UNIQUE);

    /*
    // PROBLEMATICO e DETERMINISTICO
    batch_src[0] = 3; batch_dst[0] = 2;
    batch_src[1] = 3; batch_dst[1] = 12;
    batch_src[2] = 5; batch_dst[2] = 6;
    batch_src[3] = 5; batch_dst[3] = 9;
    batch_src[4] = 7; batch_dst[4] = 1;
    */

#else
  dist_t *distances = DBFS.get_host_distance_vector();
  generateEvilBatch(batch_src, batch_dst, batch_size, distances,
                    device_graph.nV(), 2);
  delete[] distances;
#endif

#if 0
    printf("Generated edges: \n");
    for(int i = 0; i < batch_size; i++) {
        printf("\t%d -> %d\n", batch_src[i], batch_dst[i]);
    }
#endif

  // Copy to device
  cudaMemcpy(dev_batch_src, batch_src, sizeof(vert_t) * batch_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(dev_batch_dst, batch_dst, sizeof(vert_t) * batch_size,
             cudaMemcpyHostToDevice);

  // From src to dst
  HornetBatchUpdatePtr update_src_dst_ptrs{batch_size, dev_batch_src,
                                           dev_batch_dst};
  HornetBatchUpdate update_src_dst{update_src_dst_ptrs};

  // From dst to src
  HornetBatchUpdatePtr update_dst_src_ptrs{batch_size, dev_batch_dst,
                                           dev_batch_src};
  HornetBatchUpdate update_dst_src{update_dst_src_ptrs};

#if 0
    printf("=====================================\n");
    printf("Graph before and after batch         \n");

    device_graph.print();
    device_graph.insert(update_src_dst, true, true);
    printf(" --- \n");
    device_graph.print();

    printf("=====================================\n");
    printf("Inverse graph before and after batch \n");

    device_graph.print();
    device_graph_inv.insert(update_dst_src, true, true);
    printf(" --- \n");
    device_graph_inv.print();
#else
  device_graph.insert(update_src_dst, true, true);
  device_graph_inv.insert(update_dst_src, true, true);
#endif

  // =======================================================================

  DBFS.update(update_src_dst);

  printf("===================================\n");
  bool valid = DBFS.validate();
  printf("Validation result: %d\n", valid);

  printf("==================================\n");
  auto stats = DBFS.get_stats();
  std::cout << "frontier_expansions_count" << stats.frontier_expansions_count
            << "initial_frontier_size" << stats.initial_frontier_size
            << "vertex_update_time" << stats.vertex_update_time
            << "expansion_time" << stats.expansion_time << std::endl;

  return 0;
}

} // namespace test

int main(int argc, char **argv) {
  int ret = 0;
  {
    // ?
    ret = test::exec(argc, argv);
  }
  return ret;
}
