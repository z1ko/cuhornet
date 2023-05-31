/**
 * @file DBFS.cu
 * @brief Test of Dynamic BFS
*/

#include <iostream>
#include <cmath>

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

    int exec(int argc, char** argv) {

        graph::GraphStd<vert_t, vert_t> host_graph;
        host_graph.read(argv[1]);

        HornetInit graph_init{host_graph.nV(), host_graph.nE(), host_graph.csr_out_offsets(), host_graph.csr_out_edges()};
        HornetGraph device_graph{graph_init};
        
        // Insert batch to increment size
        int batch_size = std::stoi(argv[2]);
        
        vert_t* batch_src = new vert_t[batch_size];
        vert_t* batch_dst = new vert_t[batch_size];

        timer::Timer<timer::DEVICE> TM;

        const char SEPARATOR = '\t';
        std::cerr << "seq"        << SEPARATOR 
                  << "bfs_time"   << SEPARATOR 
                  << "bfs_level"  << std::endl;

        int benchmarks_count = std::stoi(argv[3]);
        for (int benchmark = 0; benchmark < benchmarks_count; benchmark++) {

#if 1
            generateBatch(host_graph, batch_size, batch_src, batch_dst, 
                BatchGenType::INSERT, batch_gen_property::UNIQUE);
            
            HornetBatchUpdatePtr update_ptr{batch_size, batch_src, batch_dst};
            HornetBatchUpdate update{update_ptr};
            update.sort();

            device_graph.insert(update, true, true);
#endif

            BfsTopDown2<HornetGraph> BFS(device_graph);
            BFS.set_parameters(device_graph.max_degree_id());

            TM.start();
            BFS.run();
            TM.stop();
    
            // Remove batch of the direct edges
            auto update_soa_ptr = update.in_edge().get_soa_ptr();
            HornetBatchUpdatePtr update_erase_ptr{update.size(), 
              update_soa_ptr.template get<0>(), update_soa_ptr.template get<1>()};

            HornetBatchUpdate update_erase{update_erase_ptr};
            update_erase.sort();

            device_graph.erase(update_erase); 

            int levels = BFS.getLevels();
            float duration = TM.duration();

            std::cerr << benchmark << SEPARATOR
                      << duration  << SEPARATOR
                      << levels    << std::endl;

            BFS.release();
        }

        std::cout << "BFS mean time: "          << TM.average() 
                  << "BFS standard deviation: " << TM.std_deviation() 
                  << "\n";

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
