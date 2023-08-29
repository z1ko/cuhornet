
/**
 * @file DBFS.cu
 * @brief Test of Dynamic BFS
 */

#include <args.hxx>
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

// Generate a batch that requires an expansion
void generateEvilBatch(
    int seed, vert_t *src, vert_t *dst, int batch_size, dist_t *dist,
    int dist_size /*, int delta, int minimum, int maximum*/) {

  srand(time(0) + seed);
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
  std::string filepath, batches_filepath;

  bool csv_header;
  bool relabel;
  bool verbose = false;

  std::vector<int> batch_sizes;
  int iterations;
};

bool parse_args(Args &args, int argc, char **argv) {
  args::ArgumentParser parser{"Benchmark for DBFS using stream of batches"};

  args::HelpFlag help{parser, "help", "Display help menu", {"h", "help"}};
  args::Flag csv_header{parser, "csv-header", "Output CSV header", {"header"}};
  args::Flag relabel{
      parser, "relabel", "Enable node relabeling for cache locality", {"r"}};
  args::Flag verbose{parser, "verbose", "Show more output", {"v"}};
  args::Positional<std::string> filepath{parser, "graph", "Graph filepath"};
  args::Positional<std::string> batches_filepath{parser, "batches",
                                                 "Generated batches output"};
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
  args.batches_filepath = args::get(batches_filepath);
  args.iterations = args::get(iterations);
  args.csv_header = csv_header;
  args.relabel = relabel;
  args.verbose = verbose;

  for (int batch_size : args::get(batch_sizes))
    args.batch_sizes.push_back(batch_size);

  return true;
}

FILE *file_open(const char *file_path, const char *how) {

  FILE *f = fopen(file_path, how);
  if (!f) {
    goto error;
  }

  return f;

error:
  if (f)
    fclose(f);
  return NULL;
}

struct Stats {
  float dbfs_time;
  float bfs_time;
  int bfs_max_level;
  int batch_size;
  float batch_generation_time;
};

// Step of the benchmark
// - Sort graph to be cache coherent to source node
// - Initialize DBFS
// - Generate a random batch
// - Update graph with batch
// - Run DBFS and get time
// - Randomize graph to simulate a new state
// - Run BFS and get time

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

  // Open a binary file to write all batches
  FILE *batches_write = file_open(args.batches_filepath.c_str(), "w");
  if (!batches_write) {
    std::cerr << "Unable to open batches file: " << args.batches_filepath
              << "\n";
    exit(1);
  }

  timer::Timer<timer::DEVICE> device_timer;
  timer::Timer<timer::HOST> host_timer;

  HornetGraph graph{graph_init};
  int source_id = host_graph.max_out_degree_id();

  BFSRelabel<HornetGraph> relabeler;
  for (int batch_size : args.batch_sizes) {

    int allocated_batch_size = batch_size;
    if (host_graph.is_undirected())
      allocated_batch_size *= 2;

    vert_t *batch_src = new vert_t[allocated_batch_size];
    vert_t *batch_dst = new vert_t[allocated_batch_size];

    for (int benchmark = 0; benchmark < args.iterations; benchmark++) {
      Stats stats;

      if (args.verbose) {
        printf("===========================================================\n");
        printf("INTERATION: %d\n\n", benchmark);
        printf("Graph at the beginning:\n");
        // graph.print();
      }

      // ========================================================================
      // SETUP DBFS

      // Sort graph to be cache coherent in regard to source node,
      // this simulate a previous performant state.
      auto d_relabel_map = relabeler.relabel(graph, source_id, false);
      auto d_relabel_map_inv = relabeler.inverse(d_relabel_map);

      /*
      if (args.verbose) {

        printf("Relabel map:\n");
        thrust::host_vector<vert_t> h_map = d_relabel_map;
        for (unsigned int i = 0; i < (unsigned int)h_map.size(); ++i)
          printf("%2d -> %2d\n", i, h_map[i]);

        printf("Graph after the initial relabeling:\n");
        graph.print();
      }
      */

      // Initialize DBFS to sorted graph, source_id is now 0
      // Heat the cache, in a dynamic setting the cache should be hot
      DynamicBFS<HornetGraph> DBFS{graph, graph};
      for (int i = 0; i < 10; i++)
        DBFS.run();

      // ========================================================================
      // BATCH GENERATION

      // Generate a random batch
      INSTRUMENT(host_timer, stats.batch_generation_time, {
        generateEvilBatch(benchmark * batch_size, batch_src, batch_dst,
                          batch_size, nullptr, graph.nV()
                          /*, batch_delta, min_batch_level, max_batch_level*/);

        // Insert reverse edges of batch
        if (host_graph.is_undirected()) {
          memcpy(batch_src + batch_size, batch_dst,
                 sizeof(vert_t) * batch_size);
          memcpy(batch_dst + batch_size, batch_src,
                 sizeof(vert_t) * batch_size);
        }
      });

      BatchUpdatePtrHost update_data_ptr{allocated_batch_size, batch_src,
                                         batch_dst};
      BatchUpdate update{update_data_ptr};
      if (args.verbose) {
        printf("Initial update batch:\n");
        update.print();
      }

      // ========================================================================
      // BATCH INSERTION

      graph.insert(update, true, true);
      if (args.verbose) {
        printf("Real update batch:\n");
        update.print();
        printf("Graph after update:\n");
        // graph.print();
      }

      // ========================================================================
      // SAVE UPDATE BATCH

      std::stringstream buff;
      buff << graph_name << "-" << batch_size << "-" << benchmark;
      std::string transient = buff.str();

      // Obtain a batch using the base node ids
      BatchUpdate update_base = relabeler.apply(update, d_relabel_map_inv);
      auto update_base_soa = update_base.in_edge().get_soa_ptr();

      if (args.verbose) {
        printf("Batch update that will be saved\n");
        update_base.print();
      }

      // Describes the batch
      fprintf(batches_write, "%s %d %d\n", transient.c_str(),
              update_base.size(), host_graph.is_undirected());

      thrust::device_vector<vid_t> d_src(update_base.size());
      thrust::device_vector<vid_t> d_dst(update_base.size());
      thrust::copy(update_base_soa.template get<0>(),
                   update_base_soa.template get<0>() + update_base.size(),
                   d_src.begin());
      thrust::copy(update_base_soa.template get<1>(),
                   update_base_soa.template get<1>() + update_base.size(),
                   d_dst.begin());

      thrust::host_vector<vert_t> h_src = d_src;
      thrust::host_vector<vert_t> h_dst = d_dst;

      if (args.verbose) {
        printf("Batch update that will be saved\n");
        for (int i = 0; i < update_base.size(); ++i)
          printf("%2d -> %2d\n", h_src[i], h_dst[i]);
      }

      // Write all batch edges
      for (int i = 0; i < update_base.size(); ++i)
        fprintf(batches_write, "%d %d\n", h_src[i], h_dst[i]);

      // ========================================================================
      // DBFS BENCHMARK

      // Apply dynamic BFS update
      INSTRUMENT(device_timer, stats.dbfs_time, DBFS.update(update));
      bool valid = DBFS.validate();
      auto dbfs_stats = DBFS.get_stats();

      // ========================================================================
      // BFS BENCHMARK

      // Apply random ordering of the nodes to simulate a dirty initial state
      auto d_random_map = relabeler.randomize(graph);
      auto d_random_map_inv = relabeler.inverse(d_random_map);

      thrust::host_vector<vert_t> h_map = d_random_map;
      /*
      if (args.verbose) {
        printf("Random map:\n");
        for (unsigned int i = 0; i < (unsigned int)h_map.size(); ++i)
          printf("%2d -> %2d\n", i, h_map[i]);
        printf("Graph after randomization:\n");
        graph.print();
      }
      */

      BfsTopDown2<HornetGraph> BFS{graph};
      BFS.set_parameters(h_map[0]);

      INSTRUMENT(device_timer, stats.bfs_time, BFS.run());
      stats.bfs_max_level = BFS.getLevels();

      // ========================================================================
      // BATCH REMOVAL

      BatchUpdate update_erase = relabeler.apply(update, d_random_map);
      if (args.verbose) {
        printf("Delete batch after mapping:\n");
        update_erase.print();
      }

      graph.erase(update_erase);
      assert(graph.nE() == host_graph.nE() && "Edges count is shifting!");

      // Revert relabeling of the graph to get consistent output...
      relabeler.apply(graph, d_random_map_inv);
      relabeler.apply(graph, d_relabel_map_inv);

      // ========================================================================
      // OUTPUT

      if (valid) {
        std::cout << graph_name << SEP << transient << SEP << batch_size << SEP
                  << benchmark << SEP << dbfs_stats.frontier_expansions_count
                  << SEP << dbfs_stats.bfs_stats.frontier_expansions_count
                  << SEP << dbfs_stats.initial_frontier_size << SEP
                  << dbfs_stats.vertex_update_time << SEP
                  << dbfs_stats.expansion_time << SEP
                  << dbfs_stats.min_frontier_size << SEP
                  << dbfs_stats.median_frontier_size << SEP
                  << dbfs_stats.max_frontier_size << SEP
                  << dbfs_stats.bfs_stats.max_frontier_size << SEP
                  << stats.dbfs_time << SEP << stats.bfs_time << SEP
                  << stats.bfs_max_level << SEP << stats.batch_generation_time
                  << std::endl;
      } else {
        std::cerr << "Error...\n";
      }

      DBFS.release();
    }

    delete[] batch_src;
    delete[] batch_dst;
  }

  fclose(batches_write);
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
