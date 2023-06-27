/**
 * @file DBFS.cu
 * @brief Test of Dynamic BFS
 */

#include <cmath>
#include <fstream>
#include <iostream>

#include <Graph/GraphStd.hpp>
#include <StandardAPI.hpp>
#include <Util/BatchFunctions.hpp>
#include <Util/CommandLineParam.hpp>

#include <Dynamic/DynamicBFS/DBFS.cuh>
#include <Static/BreadthFirstSearch/TopDown2.cuh>
#include <Static/ClusteringCoefficient/cc.cuh>

namespace test {

using namespace hornets_nest;
using namespace graph::structure_prop;
using namespace graph::parsing_prop;

using HornetInit = ::hornet::HornetInit<vert_t>;
using HornetGraph = ::hornet::gpu::Hornet<vert_t>;
using HornetBatchUpdatePtr =
    hornet::BatchUpdatePtr<vert_t, hornet::EMPTY, hornet::DeviceType::HOST>;
using HornetBatchUpdate = hornet::gpu::BatchUpdate<vert_t>;

// Generate a batch that requires an expansion
void generateEvilBatch(vert_t *src, vert_t *dst, int batch_size, dist_t *dist,
                       int dist_size, int delta, int minimum, int maximum) {
  srand(time(0));
  for (int i = 0; i < batch_size; i++) {

    vert_t src_id = rand() % dist_size;
    vert_t dst_id = rand() % dist_size;

    /*
    while(dist[src_id] == INF || dist[dst_id] - dist[src_id] < delta) {

        // Force minimum level of the edge source
        if (minimum != -1 && dist[src_id] < minimum)
          continue;

        // Force maximum level of edge source
        //if (maximum != -1 && dist[src_id] > maximum)
        //  continue;

        src_id = rand() % dist_size;
        dst_id = rand() % dist_size;
    }
    */

    // Force minimum level of the edge source
    // if (minimum != -1 && dist[src_id] < minimum)
    //  continue;

    src[i] = src_id;
    dst[i] = dst_id;
  }
}

int exec(int argc, char **argv) {
  std::cout << "Args: <graph> <batch_size_limit> <benchmark_count> "
               "<batch_delta> <minimum_batch_level> <maximum_batch_level>"
            << std::endl;

  graph::GraphStd<vert_t, vert_t> host_graph(ENABLE_INGOING);
  host_graph.read(argv[1]);

  int batch_size_limit = std::stoi(argv[2]);
  int benchmarks_count = std::stoi(argv[3]);

  // Minimum delta in batch update
  int batch_delta = 1;
  if (argc >= 4)
    batch_delta = std::stoi(argv[4]);

  // Minimum level of batch update
  int minimum_batch_level = -1;
  if (argc >= 5)
    minimum_batch_level = std::stoi(argv[5]);

  // Maximum level of batch update
  int maximum_batch_level = -1;
  if (argc >= 6)
    maximum_batch_level = std::stoi(argv[6]);

  std::cout << "Benchmark"
            << "\n\tgraph: " << argv[1]
            << "\n\tbatch_size_limit: " << batch_size_limit
            << "\n\tbenchmarks_count: " << benchmarks_count
            << "\n\tbatch_delta: " << batch_delta
            << "\n\tminimum_batch_level: " << minimum_batch_level
            << "\n\tmaximum_batch_level: " << maximum_batch_level << std::endl;

  // CSV Header
  char SEPARATOR = '\t';
  std::cerr << "graph" << SEPARATOR << "seq" << SEPARATOR
            << "frontier_expansions_count" << SEPARATOR
            << "initial_frontier_size" << SEPARATOR << "vertex_update_time"
            << SEPARATOR << "expansion_time" << SEPARATOR << "dbfs_time"
            << SEPARATOR << "bfs_time" << SEPARATOR << "bfs_max_level"
            << SEPARATOR << "batch_size" << SEPARATOR << "batch_generation_time"
            << SEPARATOR << "global_clustering_coeff" << SEPARATOR
            << "triangles" << SEPARATOR << "density" << SEPARATOR << "speedup"
            << std::endl;

  HornetInit graph_init{host_graph.nV(), host_graph.nE(),
                        host_graph.csr_out_offsets(),
                        host_graph.csr_out_edges()};
  HornetGraph device_graph{graph_init};

  timer::Timer<timer::DEVICE> device_timer;
  timer::Timer<timer::HOST> host_timer;

  int batch_sizes[] = {100,   500,    1000,   5000,    10000,
                       50000, 100000, 500000, 1000000, 5000000};
  int batch_sizes_count = 10;

  for (int i = 0; i < batch_sizes_count; i++) {

    int batch_size = batch_sizes[i];
    if (batch_size > batch_size_limit)
      break;

    int allocated_batch_size = batch_size;
    if (host_graph.is_undirected())
      allocated_batch_size *= 2;

    vert_t *batch_src = new vert_t[allocated_batch_size];
    vert_t *batch_dst = new vert_t[allocated_batch_size];

    for (int benchmark = 0; benchmark < benchmarks_count; benchmark++) {
      std::cout << "batch_size: " << batch_size << ", benchmark: " << benchmark
                << std::endl;

      DynamicBFS<HornetGraph> DBFS{device_graph, device_graph};
      DBFS.set_source(device_graph.max_degree_id());
      DBFS.run();

      host_timer.start();
      generateEvilBatch(batch_src, batch_dst, batch_size, nullptr,
                        device_graph.nV(), batch_delta, minimum_batch_level,
                        maximum_batch_level);

      // Insert reverse edges of batch
      if (host_graph.is_undirected()) {
        memcpy(batch_src + batch_size, batch_dst, sizeof(vert_t) * batch_size);
        memcpy(batch_dst + batch_size, batch_src, sizeof(vert_t) * batch_size);
      }

      host_timer.stop();

      // Insert direct edges
      HornetBatchUpdate update{
          HornetBatchUpdatePtr{allocated_batch_size, batch_src, batch_dst}};
      device_graph.insert(update, true, true);

      // Apply dynamic BFS update
      device_timer.start();
      DBFS.update(update);
      device_timer.stop();

      bool valid = DBFS.validate();
      auto stats = DBFS.get_stats();

      // Calculate new graph metrics
      ClusteringCoefficient clustering{device_graph};
      // clustering.init();
      // clustering.run();

      // VERY SLOW FOR SOME GRAPHS
      auto gcc = 0.0; // clustering.getGlobalClusteringCoeff();
      auto tcn = 0;   // clustering.countTriangles();
      if (host_graph.is_undirected())
        tcn /= 2;

      float density = (float)device_graph.nE() /
                      (device_graph.nV() * (device_graph.nV() - 1));
      if (host_graph.is_undirected())
        density *= 2.0f;

      // Remove batch of the direct edges
      auto update_soa_ptr = update.in_edge().get_soa_ptr();
      HornetBatchUpdatePtr update_erase_ptr{update.size(),
                                            update_soa_ptr.template get<0>(),
                                            update_soa_ptr.template get<1>()};

      HornetBatchUpdate update_erase{update_erase_ptr};
      device_graph.erase(update_erase);

      if (valid) {
        // CSV Data
        auto dbfs_time = device_timer.duration();
        std::cerr << argv[1] << SEPARATOR << benchmark << SEPARATOR
                  << stats.frontier_expansions_count << SEPARATOR
                  << stats.initial_frontier_size << SEPARATOR
                  << stats.vertex_update_time << SEPARATOR
                  << stats.expansion_time << SEPARATOR << dbfs_time << SEPARATOR
                  << stats.bfs_time << SEPARATOR << stats.bfs_max_level
                  << SEPARATOR << batch_size << SEPARATOR
                  << host_timer.duration() << SEPARATOR << gcc << SEPARATOR
                  << tcn << SEPARATOR << density << SEPARATOR
                  << (stats.bfs_time / dbfs_time) << std::endl;
      } else {
        benchmark -= 1;
        std::cout << "Error...\n";
      }

      DBFS.release();
    }

    delete[] batch_src;
    delete[] batch_dst;
  }

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
