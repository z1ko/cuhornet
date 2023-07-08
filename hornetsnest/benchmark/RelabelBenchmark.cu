
#include <Dynamic/DynamicBFS/DBFS.cuh>
#include <Graph/GraphStd.hpp>
#include <StandardAPI.hpp>
#include <Static/BreadthFirstSearch/TopDown2.cuh>
#include <Transform/Relabeling/BFSRelabel.cuh>
#include <Util/BatchFunctions.hpp>
#include <Util/CommandLineParam.hpp>
#include <args.hxx>
#include <cstdio>
#include <cstdlib>
#include <iostream>

struct Args {
  std::string filepath;
  int iterations = 10;
  int heating_iterations = 10;
  bool csv_header = false;
  bool verbose = false;
};

bool parse_args(Args &args, int argc, char **argv) {

  args::ArgumentParser parser{"Benchmark for cache relabeling functionality"};
  args::HelpFlag help{parser, "help", "Display help menu", {"h", "help"}};
  args::Flag csv_header{parser, "csv-header", "Output CSV header", {"header"}};
  args::Flag verbose{parser, "verbose", "Output more informations", {"v"}};
  args::ValueFlag<int> hiters{parser,
                              "cache-heating-iters",
                              "How many cache heating iterations",
                              {"-chi"}};
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
  args.verbose = verbose;
  if (hiters)
    args.heating_iterations = args::get(hiters);

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

#define BFS_SETUP(bfs, source)                                                 \
  {                                                                            \
    bfs.reset();                                                               \
    bfs.set_parameters(source);                                                \
  }
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

  timer::Timer<timer::DEVICE> dev_timer;
  graph::GraphStd<vert_t, vert_t> host_graph(ENABLE_INGOING);
  host_graph.read(args.filepath.c_str());
  HornetInit graph_init{host_graph.nV(), host_graph.nE(),
                        host_graph.csr_out_offsets(),
                        host_graph.csr_out_edges()};

  const int nE = host_graph.nE();
  const int nV = host_graph.nV();
  float average_degree = (float)nE / nV;
  float density = (float)nE / ((float)nV * ((float)nV - 1));
  if (host_graph.is_undirected())
    density *= 2.0f;

  // graph - bfs_time - bfs_time_rel - bfs_time_rel_ord - map_time_rel -
  // map_time_rel_ord
  const char SEP = '\t';
  if (args.csv_header) {
    std::cerr << "graph" << SEP << "bfs_time" << SEP << "bfs_time_rel" << SEP
              << "bfs_time_rel_ord" << SEP << "map_time_rel" << SEP
              << "map_time_rel_ord" << SEP << "avg_degree" << SEP << "density"
              << SEP << "vertices" << SEP << "edges" << std::endl;
  }

  HornetGraph graph{graph_init};

  BFSRelabel<HornetGraph> relabeler{graph};
  for (int i = 0; i < args.iterations; i++) {
    float bfs_time, bfs_time_rel, bfs_time_rel_ord, map_time_rel,
        map_time_rel_ord;

    if (args.verbose) {
      printf("Graph before randomize:\n\n");
      graph.print();
    }

    // Apply a random permutation to the graph
    auto d_map = relabeler.randomize();

    if (args.verbose) {
      printf("Mapping (from -> to):\n\n");
      thrust::host_vector<dist_t> h_map = d_map;
      for (unsigned int i = 0; i < (unsigned int)h_map.size(); ++i) {
        printf("%4d -> %4d\n", i, h_map[i]);
      }
      printf("Graph after randomize:\n\n");
      graph.print();
    }

    // Run a normal BFS
    BfsTopDown2<HornetGraph> bfs{graph};
    vert_t source = rand() % host_graph.nV();

    // A few runs to heat the cache
    for (int i = 0; i < args.heating_iterations; ++i) {
      BFS_SETUP(bfs, source);
      bfs.run();
    }

    BFS_SETUP(bfs, source);
    INSTRUMENT(dev_timer, bfs_time, bfs.run());

    // Apply reordering for current source
    d_map = relabeler.relabel(source, false);
    auto stats = relabeler.get_stats();
    map_time_rel = stats.generation_time + stats.applying_time;

    // Run new bfs, source is now at id zero
    BfsTopDown2<HornetGraph> rbfs{graph};

    // A few runs to heat the cache
    for (int i = 0; i < args.heating_iterations; ++i) {
      BFS_SETUP(rbfs, 0);
      rbfs.run();
    }

    BFS_SETUP(rbfs, 0);
    INSTRUMENT(dev_timer, bfs_time_rel, rbfs.run());

    // Apply reordering with sorting
    d_map = relabeler.relabel(0, true);
    stats = relabeler.get_stats();
    map_time_rel_ord = stats.generation_time + stats.applying_time;

    // Run new bfs, source is now at id zero
    BfsTopDown2<HornetGraph> robfs{graph};

    // A few runs to heat the cache
    for (int i = 0; i < args.heating_iterations; ++i) {
      BFS_SETUP(robfs, 0);
      robfs.run();
    }

    BFS_SETUP(robfs, 0);
    INSTRUMENT(dev_timer, bfs_time_rel_ord, robfs.run());

    // Output as CSV row
    std::cerr << args.filepath << SEP << bfs_time << SEP << bfs_time_rel << SEP
              << bfs_time_rel_ord << SEP << map_time_rel << SEP
              << map_time_rel_ord << SEP << average_degree << SEP << density
              << SEP << graph.nV() << SEP << graph.nE() << std::endl;
  }

  return 0;
}
