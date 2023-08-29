/**
 * @file DBFS.cu
 * @brief Test of Dynamic BFS
 */

#include <args.hxx>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <Graph/GraphStd.hpp>
#include <StandardAPI.hpp>
#include <Util/BatchFunctions.hpp>
#include <Util/CommandLineParam.hpp>

#include <Dynamic/DynamicBFS/DBFS.cuh>
#include <Static/BreadthFirstSearch/TopDown2.cuh>
#include <Transform/Relabeling/BFSRelabel.cuh>

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

using HornetInit = ::hornet::HornetInit<vert_t>;
using HornetGraph = ::hornet::gpu::Hornet<vert_t>;
using BatchUpdatePtrHost =
    hornet::BatchUpdatePtr<vert_t, hornet::EMPTY, hornet::DeviceType::HOST>;
using BatchUpdatePtrDevice =
    hornet::BatchUpdatePtr<vert_t, hornet::EMPTY, hornet::DeviceType::DEVICE>;
using BatchUpdate = hornet::gpu::BatchUpdate<vert_t>;

struct Args {
  std::string filepath;
  std::vector<int> batch_sizes;
  int iterations;
  bool verbose;
};

bool parse_args(Args &args, int argc, char **argv) {
  args::ArgumentParser parser{"Benchmark for DBFS using stream of batches"};

  args::HelpFlag help{parser, "help", "Display help menu", {"h", "help"}};
  args::Flag verbose{parser, "verbose", "Show more output", {"v"}};
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
  args.verbose = verbose;

  for (int batch_size : args::get(batch_sizes))
    args.batch_sizes.push_back(batch_size);

  return true;
}

int exec(int argc, char **argv) {

  Args args;
  if (!parse_args(args, argc, argv))
    return 1;

  std::string graph_name = args.filepath;
  auto b = graph_name.find_last_of('/') + 1;
  auto e = graph_name.find_last_of('.');
  graph_name = graph_name.substr(b, e);

  graph::GraphStd<vert_t, vert_t> host_graph(ENABLE_INGOING);
  host_graph.read(args.filepath.c_str(), graph::parsing_prop::NONE);
  HornetInit graph_init{host_graph.nV(), host_graph.nE(),
                        host_graph.csr_out_offsets(),
                        host_graph.csr_out_edges()};

  assert(host_graph.is_undirected() && "GRAPH MUST BE UNDIRECTED!\n");

#if 0
  // CSV Header
  const char SEP = '\t';
  if (args.csv_header) {
    std::cout << "graph" << SEP << "transient" << SEP << "batch_size" << SEP
              << "seq" << SEP << "dbfs_frontier_expansions_count" << SEP
              << "bfs_frontier_expansion_count" << SEP
              << "initial_frontier_size" << SEP << "vertex_update_time" << SEP
              << "expansion_time" << SEP << "min_frontier_size" << SEP
              << "median_frontier_size" << SEP << "dbfs_max_frontier_size"
              << SEP << "bfs_max_frontier_size" << SEP << "dbfs_time" << SEP
              << "bfs_time" << SEP << "bfs_max_level" << SEP
              << "batch_generation_time" << std::endl;
  }
#endif
#if 0
  // Open a binary file to write all batches
  FILE *batches_write = file_open(args.batches_filepath.c_str(), "w");
  if (!batches_write) {
    std::cerr << "Unable to open batches file: " << args.batches_filepath
              << "\n";
    exit(1);
  }
#endif

  timer::Timer<timer::DEVICE> device_timer;
  timer::Timer<timer::HOST> host_timer;

  int source_id = host_graph.max_out_degree_id();

  for (int batch_size : args.batch_sizes) {

    int allocated_batch_size = batch_size;
    if (host_graph.is_undirected())
      allocated_batch_size *= 2;

    vert_t *batch_src = new vert_t[allocated_batch_size];
    vert_t *batch_dst = new vert_t[allocated_batch_size];

    for (int benchmark = 0; benchmark < args.iterations; benchmark++) {

      HornetGraph graph{graph_init};
      DynamicBFS<HornetGraph> DBFS{graph, graph};
      DBFS.set_source(source_id);
      DBFS.run();

      // Generate batch edges
      generateBatch(host_graph, batch_size, batch_src, batch_dst,
                    hornets_nest::BatchGenType::REMOVE);

      // NB: graph must be undirected
      memcpy(batch_src + batch_size, batch_dst, sizeof(vert_t) * batch_size);
      memcpy(batch_dst + batch_size, batch_src, sizeof(vert_t) * batch_size);

      BatchUpdatePtrHost batch_ptr{allocated_batch_size, batch_src, batch_dst};
      BatchUpdate batch{batch_ptr};

      if (args.verbose) {
        printf("Batch pre erase:\n");
        batch.print();
      }

      graph.erase(batch);

      float dbfs_time = 0.0f;
      BatchUpdate batch_copy{batch_ptr};
      INSTRUMENT(device_timer, dbfs_time, DBFS.erase(batch_copy, args.verbose));

      if (!DBFS.validate()) {
        printf("ERROR!\n");
      }

      BfsTopDown2<HornetGraph> BFS{graph};
      BFS.set_parameters(source_id);

      float bfs_time = 0.0f;
      INSTRUMENT(device_timer, bfs_time, BFS.run());

      printf("speedup: %f\n", bfs_time / dbfs_time);
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
