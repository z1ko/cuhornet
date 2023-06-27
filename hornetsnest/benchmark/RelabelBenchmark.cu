
#include <Dynamic/DynamicBFS/DBFS.cuh>
#include <Graph/GraphStd.hpp>
#include <StandardAPI.hpp>
#include <Static/BreadthFirstSearch/TopDown2.cuh>
#include <Util/BatchFunctions.hpp>
#include <Util/CommandLineParam.hpp>
#include <args.hxx>
#include <cstdlib>
#include <iostream>

struct Args {
  std::string filepath;
  int iterations = 10;
  bool csv_header = false;
};

bool parse_args(Args &args, int argc, char **argv) {

  args::ArgumentParser parser{"Benchmark for cache relabeling functionality"};
  args::HelpFlag help{parser, "help", "Display help menu", {"h", "help"}};
  args::Flag csv_header{parser, "csv-header", "Output CSV header", {"header"}};
  args::Positional<std::string> filepath{parser, "graph",
                                         "Filepath of the graph"};
  args::Positional<int> iterations{parser, "iterations",
                                   "How many benchmarks iterations"};

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
  args.csv_header = csv_header;

  return true;
}

using namespace hornets_nest;
using namespace graph::structure_prop;
using namespace graph::parsing_prop;

using HornetInit = ::hornet::HornetInit<vert_t>;
using HornetGraph = ::hornet::gpu::Hornet<vert_t>;
using HornetBatchUpdatePtr =
    hornet::BatchUpdatePtr<vert_t, hornet::EMPTY, hornet::DeviceType::HOST>;
using HornetBatchUpdate = hornet::gpu::BatchUpdate<vert_t>;

#define INSTRUMENT(timer, result, expr)                                        \
  {                                                                            \
    timer.start();                                                             \
    expr;                                                                      \
    timer.stop();                                                              \
    result = timer.duration();                                                 \
  }

int main(int argc, char **argv) {

  Args args;
  if (!parse_args(args, argc, argv))
    return 1;

  graph::GraphStd<vert_t, vert_t> host_graph(ENABLE_INGOING);
  host_graph.read(args.filepath.c_str());
  HornetInit graph_init{host_graph.nV(), host_graph.nE(),
                        host_graph.csr_out_offsets(),
                        host_graph.csr_out_edges()};

  const char SEP = '\t';
  if (args.csv_header) {
    std::cerr << "graph" << SEP << "base_bfs_time" << SEP
              << "basic_relabel_bfs_time" << SEP << "basic_relabel_time" << SEP
              << "edges_relabel_bfs_time" << SEP << "edges_relabel_time"
              << std::endl;
  }

  timer::Timer<timer::DEVICE> dev_timer;
  for (int i = 0; i < args.iterations; i++) {
    vert_t source = rand() % host_graph.nV();

    float base_bfs_time;
    {
      HornetGraph graph{graph_init};
      BfsTopDown2<HornetGraph> BFS{graph};
      BFS.set_parameters(source);

      INSTRUMENT(dev_timer, base_bfs_time, BFS.run());
    }

    float basic_relabel_time;
    float basic_relabel_bfs_time;
    {
      HornetGraph graph{graph_init};
      DynamicBFS<HornetGraph> DBFS{graph, graph};
      DBFS.set_source(source);
      DBFS.run();

      INSTRUMENT(dev_timer, basic_relabel_time,
                 DBFS.apply_cache_reordering(false));

      BfsTopDown2<HornetGraph> BFS{graph};
      BFS.set_parameters(0);

      INSTRUMENT(dev_timer, basic_relabel_bfs_time, BFS.run());
    }

    float edges_relabel_time;
    float edges_relabel_bfs_time;
    {
      HornetGraph graph{graph_init};
      DynamicBFS<HornetGraph> DBFS{graph, graph};
      DBFS.set_source(source);
      DBFS.run();

      INSTRUMENT(dev_timer, edges_relabel_time,
                 DBFS.apply_cache_reordering(true));

      BfsTopDown2<HornetGraph> BFS{graph};
      BFS.set_parameters(0);

      INSTRUMENT(dev_timer, edges_relabel_bfs_time, BFS.run());
    }

    std::cerr << args.filepath << SEP << base_bfs_time << SEP
              << basic_relabel_bfs_time << SEP << basic_relabel_time << SEP
              << edges_relabel_bfs_time << SEP << edges_relabel_time
              << std::endl;
  }

  return 0;
}
