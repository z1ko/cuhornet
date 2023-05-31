/**
 * @file DBFS.cu
 * @brief Test of Dynamic BFS
*/

#include <iostream>
#include <cmath>
#include <iostream>
#include <fstream>

#include <StandardAPI.hpp>
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <Util/BatchFunctions.hpp>

#include <Static/BreadthFirstSearch/TopDown2.cuh>
#include <Dynamic/DynamicBFS/DBFS.cuh>

namespace test {

    using namespace hornets_nest;
    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;

    using HornetInit = ::hornet::HornetInit<vert_t>;
    using HornetGraph = ::hornet::gpu::Hornet<vert_t>;
    using HornetBatchUpdatePtr = hornet::BatchUpdatePtr<vert_t, hornet::EMPTY, hornet::DeviceType::HOST>;
    using HornetBatchUpdate = hornet::gpu::BatchUpdate<vert_t>;

    // Generate a batch that requires an expansion
    void generateEvilBatch(vert_t* src, vert_t* dst, int batch_size, dist_t* dist, int dist_size, int delta, int minimum, int maximum) {
        srand(time(0));
        for(int i = 0; i < batch_size; i++) {

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
            //if (minimum != -1 && dist[src_id] < minimum)
            //  continue;

            src[i] = src_id;
            dst[i] = dst_id;
        }
    }


    int exec(int argc, char** argv) {
        std::cout << "Args: <graph> <batch_size_limit> <benchmark_count> <batch_delta> <minimum_batch_level> <maximum_batch_level>"
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
                  << "\n\tgraph: "                << argv[1]
                  << "\n\tbatch_size_limit: "     << batch_size_limit
                  << "\n\tbenchmarks_count: "     << benchmarks_count
                  << "\n\tbatch_delta: "          << batch_delta
                  << "\n\tminimum_batch_level: "  << minimum_batch_level
                  << "\n\tmaximum_batch_level: "  << maximum_batch_level
                  << std::endl;

        // CSV Header
        char SEPARATOR = '\t';
        std::cerr << "seq"                        << SEPARATOR
                  << "frontier_expansions_count"  << SEPARATOR
                  << "initial_frontier_size"      << SEPARATOR
                  << "vertex_update_time"         << SEPARATOR
                  << "expansion_time"             << SEPARATOR
                  << "dbfs_time"                  << SEPARATOR
                  << "bfs_time"                   << SEPARATOR
                  << "bfs_max_level"              << SEPARATOR
                  << "batch_size"                 << SEPARATOR
                  << "batch_generation_time"      << std::endl;

        HornetInit graph_init{host_graph.nV(), host_graph.nE(), host_graph.csr_out_offsets(), host_graph.csr_out_edges()};
        HornetGraph device_graph{graph_init};

        // Use only UNDIRECTED GRAPHS
        //HornetInit graph_init_inv{host_graph.nV(), host_graph.nE(), host_graph.csr_in_offsets(), host_graph.csr_in_edges()};
        //HornetGraph device_graph_inv{graph_init_inv};

        BfsTopDown2<HornetGraph> BFS(device_graph);
        BFS.set_parameters(device_graph.max_degree_id());
        BFS.run();

        // Base distance vector used to generate smart batches
        const auto nV = device_graph.nV();
        auto* base_distances = new dist_t[nV];
        gpu::copyToHost(BFS.get_distance_vector(), nV, base_distances);
        BFS.release();

        timer::Timer<timer::DEVICE> device_timer;
        timer::Timer<timer::HOST> host_timer;

        int batch_sizes[] = { 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000 };
        int batch_sizes_count = 10;

        for (int i = 0; i < batch_sizes_count; i++) {
          
          int batch_size = batch_sizes[i];
          if (batch_size > batch_size_limit)
            break;

          vert_t* batch_src = new vert_t[batch_size];
          vert_t* batch_dst = new vert_t[batch_size];

          DynamicBFS<HornetGraph> DBFS{device_graph, device_graph /*device_graph_inv*/, 5000000};
          for (int benchmark = 0; benchmark < benchmarks_count; benchmark++) {
            std::cout << "batch_size: "  << batch_size << ", benchmark: " << benchmark
                      << std::endl;

            DBFS.reset();
            DBFS.set_source(device_graph.max_degree_id());
            DBFS.run();

            host_timer.start();
            generateEvilBatch(batch_src, batch_dst, batch_size, base_distances, device_graph.nV(), 
                batch_delta, minimum_batch_level, maximum_batch_level);

            host_timer.stop();

            // Insert direct edges
            HornetBatchUpdate update{HornetBatchUpdatePtr{batch_size, batch_src, batch_dst}};
            device_graph.insert(update, true, true); 

            // Insert parent edges
            HornetBatchUpdate update_inv{HornetBatchUpdatePtr{batch_size, batch_dst, batch_src}};
            //device_graph_inv.insert(update_inv, true, true);
            device_graph.insert(update_inv, true, true);

            // Apply dynamic BFS update
            device_timer.start();
            DBFS.update(batch_dst, batch_size);
            device_timer.stop();

            bool valid = DBFS.validate();
            auto stats = DBFS.get_stats();

            // Remove batch of the direct edges
            auto update_soa_ptr = update.in_edge().get_soa_ptr();
            HornetBatchUpdatePtr update_erase_ptr{update.size(), 
              update_soa_ptr.template get<0>(), update_soa_ptr.template get<1>()};

            HornetBatchUpdate update_erase{update_erase_ptr};
            device_graph.erase(update_erase); 

            // Remove batch of parent edges
            auto update_inv_soa_ptr = update_inv.in_edge().get_soa_ptr();
            HornetBatchUpdatePtr update_inv_erase_ptr{update_inv.size(),
              update_inv_soa_ptr.template get<0>(), update_inv_soa_ptr.template get<1>()};

            HornetBatchUpdate update_inv_erase{update_inv_erase_ptr};
            //device_graph_inv.erase(update_inv_erase);
            device_graph.erase(update_inv_erase);

            if (valid) {
                // CSV Data
                std::cerr << benchmark                           << SEPARATOR
                          << stats.frontier_expansions_count     << SEPARATOR
                          << stats.initial_frontier_size         << SEPARATOR
                          << stats.vertex_update_time            << SEPARATOR
                          << stats.expansion_time                << SEPARATOR
                          << device_timer.duration()             << SEPARATOR
                          << stats.bfs_time                      << SEPARATOR
                          << stats.bfs_max_level                 << SEPARATOR
                          << batch_size                          << SEPARATOR
                          << host_timer.duration()               << std::endl;
            }
            else {
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

int main(int argc, char** argv) {
    int ret = 0;
    {
        // ?
        ret = test::exec(argc, argv);
    }
    return ret;
}
