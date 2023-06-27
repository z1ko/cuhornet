
/**
 * @file DBFS.cu
 * @brief Test of Dynamic BFS
 */

#include <args.hxx>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include <Graph/GraphStd.hpp>
#include <StandardAPI.hpp>
#include <Util/BatchFunctions.hpp>
#include <Util/CommandLineParam.hpp>

#include <Dynamic/DynamicBFS/DBFS.cuh>
#include <Static/BetweennessCentrality/bc.cuh>
#include <Static/BreadthFirstSearch/TopDown2.cuh>
#include <Static/ClusteringCoefficient/cc.cuh>

#define INSTRUMENT(timer, result, expr)                                        \
  {                                                                            \
    timer.start();                                                             \
    expr;                                                                      \
    timer.stop();                                                              \
    result = timer.duration();                                                 \
  }

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
void generateEvilBatch(
    vert_t *src, vert_t *dst, int batch_size, dist_t *dist,
    int dist_size /*, int delta, int minimum, int maximum*/) {

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

struct Args {
  std::string filepath;
  bool betweenness_centrality;
  bool csv_header;
  bool relabel;

  std::vector<int> batch_sizes;
  int iterations;
};

bool parse_args(Args &args, int argc, char **argv) {
  args::ArgumentParser parser{"Benchmark for DBFS using stream of batches"};

  args::HelpFlag help{parser, "help", "Display help menu", {"h", "help"}};
  args::Flag csv_header{parser, "csv-header", "Output CSV header", {"header"}};
  args::Flag relabel{
      parser, "relabel", "Enable node relabeling for cache locality", {"r"}};
  args::Flag betweenness_centrality{parser,
                                    "betweenness_centrality",
                                    "Use the most central node according to "
                                    "the Betweenness Centrality metric.",
                                    {"bc"}};

  args::Positional<std::string> filepath{parser, "graph", "Graph filepath"};
  args::Positional<int> iterations{parser, "iters", "Iterations per batch"};
  args::PositionalList<int> batch_sizes{parser, "batch_sizes",
                                        "Sizes of the batches"};

  try {
    parser.ParseCLI(argc, argv);
  } catch (args::Help &) {
    std::cout << parser;
    return false;
  } catch (args::ParseError &e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return false;
  } catch (args::ValidationError &e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return false;
  }

  args.filepath = args::get(filepath);
  args.iterations = args::get(iterations);
  args.betweenness_centrality = betweenness_centrality;
  args.csv_header = csv_header;
  args.relabel = relabel;

  for (int batch_size : args::get(batch_sizes))
    args.batch_sizes.push_back(batch_size);

  return true;
}

int exec(int argc, char **argv) {

  Args args;
  if (!parse_args(args, argc, argv))
    return 1;

  graph::GraphStd<vert_t, vert_t> host_graph(ENABLE_INGOING);
  host_graph.read(args.filepath.c_str());
  HornetInit graph_init{host_graph.nV(), host_graph.nE(),
                        host_graph.csr_out_offsets(),
                        host_graph.csr_out_edges()};

  // CSV Header
  const char SEP = '\t';
  if (args.csv_header) {
    std::cerr << "graph" << SEP << "seq" << SEP << "frontier_expansions_count"
              << SEP << "initial_frontier_size" << SEP << "vertex_update_time"
              << SEP << "expansion_time" << SEP << "dbfs_time" << SEP
              << "bfs_time" << SEP << "bfs_max_level" << SEP << "batch_size"
              << SEP << "batch_generation_time" << SEP << "density" << SEP
              << "vertices" << SEP << "edges" << SEP << "cc" << SEP
              << "triangles" << SEP << "average_degree" << std::endl;
  }

  timer::Timer<timer::DEVICE> device_timer;
  timer::Timer<timer::HOST> host_timer;

  HornetGraph graph{graph_init};
  int source_id = host_graph.max_out_degree_id();

  // Find most central node of the graph
  // using the betweenness centrality measure
  if (args.betweenness_centrality) {

    BCCentrality BC{graph};
    BC.setRoot(host_graph.max_out_degree_id());
    BC.run();

    source_id = BC.getBestVertex();
  }

  for (int batch_size : args.batch_sizes) {

    int allocated_batch_size = batch_size;
    if (host_graph.is_undirected())
      allocated_batch_size *= 2;

    vert_t *batch_src = new vert_t[allocated_batch_size];
    vert_t *batch_dst = new vert_t[allocated_batch_size];

    for (int benchmark = 0; benchmark < args.iterations; benchmark++) {

      DynamicBFS<HornetGraph> DBFS{graph, graph};
      DBFS.set_source(source_id);
      DBFS.run();

      /*
       * TODO: Fix bug with relabeling...
       *
       * if (args.relabel)
       *  DBFS.apply_cache_reordering(false);
       */

      float batch_generation_time;
      INSTRUMENT(host_timer, batch_generation_time, {
        generateEvilBatch(batch_src, batch_dst, batch_size, nullptr, graph.nV()
                          /*, batch_delta, min_batch_level, max_batch_level*/);

        // Insert reverse edges of batch
        if (host_graph.is_undirected()) {
          memcpy(batch_src + batch_size, batch_dst,
                 sizeof(vert_t) * batch_size);
          memcpy(batch_dst + batch_size, batch_src,
                 sizeof(vert_t) * batch_size);
        }
      });

      // Insert direct edges
      HornetBatchUpdate update{
          HornetBatchUpdatePtr{allocated_batch_size, batch_src, batch_dst}};
      graph.insert(update, true, true);

      // Apply dynamic BFS update
      float dbfs_time;
      INSTRUMENT(device_timer, dbfs_time, DBFS.update(update));
      bool valid = DBFS.validate();

      // Calculate new graph metrics...
      float density =
          (float)graph.nE() / ((float)graph.nV() * ((float)graph.nV() - 1));
      if (host_graph.is_undirected())
        density *= 2.0f;

      int vertices = graph.nV();
      int edges = graph.nE();

      float average_degree = (float)graph.nE() / graph.nV();
      float clustering_coeff = 0.0f;
      int triangles = 0;

      // Dont calculate coeff for very large graphs
      if (graph.nE() <= 100'000'000) {

        ClusteringCoefficient CC{graph};
        CC.init();
        CC.run();

        clustering_coeff = CC.getGlobalClusteringCoeff();
        triangles = CC.countTriangles();
      }

      // Remove batch of the direct edges
      auto update_soa_ptr = update.in_edge().get_soa_ptr();
      HornetBatchUpdatePtr update_erase_ptr{update.size(),
                                            update_soa_ptr.template get<0>(),
                                            update_soa_ptr.template get<1>()};

      HornetBatchUpdate update_erase{update_erase_ptr};
      graph.erase(update_erase);

      if (valid) {
        auto stats = DBFS.get_stats();
        std::cerr << args.filepath << SEP << benchmark << SEP
                  << stats.frontier_expansions_count << SEP
                  << stats.initial_frontier_size << SEP
                  << stats.vertex_update_time << SEP << stats.expansion_time
                  << SEP << dbfs_time << SEP << stats.bfs_time << SEP
                  << stats.bfs_max_level << SEP << batch_size << SEP
                  << batch_generation_time << SEP << density << SEP << vertices
                  << SEP << edges << SEP << clustering_coeff << SEP << triangles
                  << SEP << average_degree << std::endl;
      } else {
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
