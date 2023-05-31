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

    // Generate a batch that requires an expansion
    void generateEvilBatch(vert_t* src, vert_t* dst, int batch_size, dist_t* dist, int dist_size, int delta, int minimum = -1) {
        srand(time(0));
        for(int i = 0; i < batch_size; i++) {

            vert_t src_id = rand() % dist_size;
            vert_t dst_id = rand() % dist_size;

            while(dist[src_id] == INF || dist[dst_id] - dist[src_id] < delta) {
                
                // Force minimum level of the edge source
                if (minimum != -1 && dist[src_id] < minimum)
                  continue;
                
                src_id = rand() % dist_size;
                dst_id = rand() % dist_size;
            }

            src[i] = src_id;
            dst[i] = dst_id;
        }
    }


    int exec(int argc, char** argv) {

        int batch_size = std::stoi(argv[2]);
        graph::GraphStd<vert_t, vert_t> host_graph(ENABLE_INGOING);
        host_graph.read(argv[1]);

        // CSV Header
        char SEPARATOR = '\t';
        std::cerr << "frontier_expansions_count"  << SEPARATOR
                  << "initial_frontier_size"      << SEPARATOR
                  << "vertex_update_time"         << SEPARATOR
                  << "expansion_time"             << SEPARATOR
                  << "dbfs_time"                  << SEPARATOR
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
        
        HornetInit graph_init{host_graph.nV(), host_graph.nE(), host_graph.csr_out_offsets(), host_graph.csr_out_edges()};
        HornetInit graph_init_inv{host_graph.nV(), host_graph.nE(), host_graph.csr_in_offsets(), host_graph.csr_in_edges()};
        
        timer::Timer<timer::DEVICE> timer;

        vert_t* batch_src = new vert_t[batch_size];
        vert_t* batch_dst = new vert_t[batch_size];

        int benchmarks_count = std::stoi(argv[3]);
        for (int benchmark = 0; benchmark < benchmarks_count; benchmark++) {
            
            HornetGraph device_graph{graph_init};
            HornetGraph device_graph_inv{graph_init_inv};

            DynamicBFS<HornetGraph> DBFS{device_graph, device_graph_inv, 5000000};
            DBFS.set_source(device_graph.max_degree_id());
            DBFS.run();

            auto *distances = DBFS.get_host_distance_vector();
            generateEvilBatch(batch_src, batch_dst, batch_size, distances, device_graph.nV(), 2);
            delete[] distances;

            HornetBatchUpdate update{HornetBatchUpdatePtr{batch_size, batch_src, batch_dst}};
            HornetBatchUpdate update_inv{HornetBatchUpdatePtr{batch_size, batch_dst, batch_src}};
            
            // Insert update into the graph
            device_graph.insert(update);
            device_graph_inv.insert(update_inv);

            // Apply dynamic BFS update
            timer.start();
            DBFS.update(batch_dst, batch_size);
            timer.stop();

            timer.print("DBFS");

            bool valid = DBFS.validate();
            auto stats = DBFS.get_stats();

            if (valid) {
                // CSV Data
                std::cerr << stats.frontier_expansions_count     << SEPARATOR
                          << stats.initial_frontier_size         << SEPARATOR
                          << stats.vertex_update_time            << SEPARATOR
                          << stats.expansion_time                << SEPARATOR
                          << timer.duration()                    << SEPARATOR
                          << std::endl;
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
