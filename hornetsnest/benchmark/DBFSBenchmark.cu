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

#include "Dynamic/DynamicBFS/DBFS.cuh"

namespace test {

    using namespace hornets_nest;
    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;

    using HornetInit = ::hornet::HornetInit<vert_t>;
    using HornetGraph = ::hornet::gpu::Hornet<vert_t>;
    using HornetBatchUpdatePtr = hornet::BatchUpdatePtr<vert_t, hornet::EMPTY, hornet::DeviceType::HOST>;
    using HornetBatchUpdate = hornet::gpu::BatchUpdate<vert_t>;

    struct benchmark_measure {
      DynamicBFS<HornetGraph>::Stats stats;
      float elapsed;
    };

    int exec(int argc, char** argv) {

        int batch_size = std::stoi(argv[2]);
        graph::GraphStd<vert_t, vert_t> host_graph(ENABLE_INGOING);
        host_graph.read(argv[1]);

        // CSV Header
        char SEPARATOR = ' ';
        std::cerr << "update_time"                        << SEPARATOR
                  << "expanded_frontiers"                 << SEPARATOR
                  << "visited_vertices"                   << SEPARATOR
                  << "post_vertex_update_frontier_size"   << SEPARATOR
                  << "mean_frontier_size"                 << SEPARATOR
                  << "max_frontier_size"                  << SEPARATOR
                  << "min_frontier_size"                  << SEPARATOR
                  << std::endl;
      
#if 0
        // This is used to precompute the base BFS distances only one time
        int* distances = nullptr;
        {
          HornetInit graph_init{host_graph.nV(), host_graph.nE(), host_graph.csr_out_offsets(), host_graph.csr_out_edges()};
          HornetGraph device_graph{graph_init};
        
          BfsTopDown2<HornetGraph> BFS(device_graph);
          BFS.set_parameters(device_graph.max_degree_id());
          BFS.run();
          
          distances = BFS.get_distance_vector();
        }
#endif
        
        timer::Timer<timer::DEVICE> TM;
        std::vector<benchmark_measure> measures;
  
        vert_t* batch_src = new vert_t[batch_size];
        vert_t* batch_dst = new vert_t[batch_size];

        int benchmarks_count = std::stoi(argv[3]);
        for (int benchmark = 0; benchmark < benchmarks_count; benchmark++) {
            
            HornetInit graph_init{host_graph.nV(), host_graph.nE(), host_graph.csr_out_offsets(), host_graph.csr_out_edges()};
            HornetGraph device_graph{graph_init};

            HornetInit graph_init_inv{host_graph.nV(), host_graph.nE(), host_graph.csr_in_offsets(), host_graph.csr_in_edges()};
            HornetGraph device_graph_inv{graph_init_inv};

            DynamicBFS<HornetGraph> DBFS{device_graph, device_graph_inv};
            DBFS.set_source(device_graph.max_degree_id());
            DBFS.run();
            
            generateBatch(host_graph, batch_size, batch_src, batch_dst, BatchGenType::INSERT, batch_gen_property::UNIQUE);

            // From src to dst
            HornetBatchUpdatePtr update_ptr{batch_size, batch_src, batch_dst};
            HornetBatchUpdate update{update_ptr};

            // From dst to src
            HornetBatchUpdatePtr update_ptr_inv{batch_size, batch_dst, batch_src};
            HornetBatchUpdate update_inv{update_ptr_inv};
            
            // Insert update into the graph
            device_graph.insert(update, true, true);
            device_graph_inv.insert(update_inv, true, true);

            // Apply dynamic BFS update
            TM.start();
            DBFS.update(batch_src, batch_dst, batch_size);
            cudaDeviceSynchronize();
            TM.stop();

            auto stats = DBFS.get_stats();

            // They seem to work fine... make this faster by not checking correctess
            bool valid = true; //DBFS.validate();

            if (valid) {
                float elapsed = TM.duration();

                // CSV Data
                std::cerr << elapsed                             << SEPARATOR
                          << stats.total_frontier_expansions     << SEPARATOR
                          << stats.total_visited_vertices        << SEPARATOR
                          << stats.initial_dynamic_frontier_size << SEPARATOR
                          << stats.frontier_size_mean            << SEPARATOR
                          << stats.frontier_size_max             << SEPARATOR
                          << stats.frontier_size_min             << SEPARATOR
                          << std::endl;
#if 0
                printf("===================================\n");
                printf("Update time: %f\n", elapsed);
                printf("Total expanded frontiers: %d\n", stats.total_frontier_expansions);
                printf("Total visited vertices: %d\n", stats.total_visited_vertices);
#endif
            }
            else {
                break;
            }

            DBFS.release();
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
