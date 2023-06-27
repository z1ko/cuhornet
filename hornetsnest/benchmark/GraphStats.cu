
#include <Graph/GraphStd.hpp>
#include <StandardAPI.hpp>
#include <Static/ClusteringCoefficient/cc.cuh>
#include <Util/BatchFunctions.hpp>
#include <Util/CommandLineParam.hpp>
#include <args.hxx>
#include <cstdlib>
#include <iostream>

#define INSTRUMENT(timer, result, expr)                                        \
  {                                                                            \
    timer.start();                                                             \
    expr;                                                                      \
    timer.stop();                                                              \
    result = timer.duration();                                                 \
  }

using namespace hornets_nest;
using namespace graph::structure_prop;
using namespace graph::parsing_prop;

using HornetInit = ::hornet::HornetInit<vert_t>;
using HornetGraph = ::hornet::gpu::Hornet<vert_t>;

struct Args {
  std::string filepath;
  bool csv_header;
};
bool parse_args(Args &args, int argc, char **argv) {

  args::ArgumentParser parser{"Provides information about a graph"};
  args::HelpFlag help{parser, "help", "Display help menu", {"h", "help"}};
  args::Flag csv_header{parser, "csv-header", "Output CSV header", {"header"}};
  args::Positional<std::string> filepath{parser, "graph",
                                         "Filepath of the graph"};

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
  args.csv_header = csv_header;

  return true;
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

  timer::Timer<timer::DEVICE> dev_timer;

  int triangles = 0;
  float cc_time = 0.0f, cc = 0.0f;
  if (host_graph.nE() <= 100'000'000) {
    // Richiede troppo tempo altrimenti

    HornetGraph graph{graph_init};
    ClusteringCoefficient CC{graph};
    CC.init();

    INSTRUMENT(dev_timer, cc_time, CC.run());
    cc = CC.getGlobalClusteringCoeff();
    triangles = CC.countTriangles();
  }
  std::string type = host_graph.is_undirected() ? "undirected" : "directed";
  float average_degree = (float)host_graph.nE() / host_graph.nV();

  float density = (float)host_graph.nE() /
                  ((float)host_graph.nV() * ((float)host_graph.nV() - 1));
  if (host_graph.is_undirected())
    density *= 2;

  // CSV Header
  const char SEP = '\t';
  if (args.csv_header) {
    std::cerr << "graph" << SEP << "cc" << SEP << "cc_time" << SEP
              << "triangles" << SEP << "density" << SEP << "vertices" << SEP
              << "edges" << SEP << "average_degree" << SEP << "type"
              << std::endl;
  }

  std::cerr << args.filepath << SEP << cc << SEP << cc_time << SEP << triangles
            << SEP << density << SEP << host_graph.nV() << SEP
            << host_graph.nE() << SEP << average_degree << SEP << type
            << std::endl;

  return 0;
}
