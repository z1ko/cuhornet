/**
 * @file DBFS.cu
 * @brief Test of Dynamic BFS
*/

#include <iostream>

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

        /**
         * Strange bug using also inverse graph, edges are not updated correctly, for now we use only undirected graphs...
         */

        int batch_size = std::stoi(argv[2]);
        graph::GraphStd<vert_t, vert_t> host_graph(UNDIRECTED);
        host_graph.read(argv[1]);

        HornetInit graph_init{host_graph.nV(), host_graph.nE(), host_graph.csr_out_offsets(), host_graph.csr_out_edges()};
        HornetGraph device_graph{graph_init};

        // Load inverse graph into the device
        //HornetInit graph_init_inv{host_graph.nV(), host_graph.nE(), host_graph.csr_in_offsets(), host_graph.csr_in_edges()};
        //HornetGraph device_graph_inv{graph_init};

        DynamicBFS<HornetGraph> DBFS{device_graph, device_graph/*device_graph_inv*/};
        DBFS.set_source(device_graph.max_degree_id());
        DBFS.run();

        timer::Timer<timer::DEVICE> TM;
        std::vector<float> times;

        vert_t* batch_src = new vert_t[batch_size];
        vert_t* batch_dst = new vert_t[batch_size];

        int benchmarks_count = std::stoi(argv[3]);
        for (int benchmark = 0; benchmark < benchmarks_count; benchmark++) {

            printf("===================================\n");
            printf("Generating batch of %d edges\n", batch_size);
            generateBatch(host_graph, batch_size, batch_src, batch_dst, BatchGenType::INSERT, batch_gen_property::UNIQUE);

            // From src to dst
            HornetBatchUpdatePtr update_src_dst_ptr{batch_size, batch_src, batch_dst};
            HornetBatchUpdate update_src_dst{update_src_dst_ptr};

            // From dst to src
            HornetBatchUpdatePtr update_dst_src_ptr{batch_size, batch_dst, batch_src};
            HornetBatchUpdate update_dst_src{update_dst_src_ptr};

            device_graph.insert(update_src_dst, true, true);
            device_graph.insert(update_dst_src, true, true);
            //sdevice_graph_inv.insert(update_dst_src, true, true);

            // Apply update
            TM.start();
            DBFS.batchUpdate(batch_src, batch_dst, batch_size);
            TM.stop();

            auto stats = DBFS.get_stats();
            bool valid = DBFS.validate();

            printf("Validation result: %d\n", valid);
            if (valid) {
                float elapsed = TM.duration();
                times.push_back(elapsed);

                printf("===================================\n");
                printf("Update time: %f\n", elapsed);
                printf("Total expanded frontiers: %d\n", stats.total_frontier_expansions);
                printf("Total visited vertices: %d\n", stats.total_visited_vertices);
            }
            else {
                break;
            }
        }

        float mean = 0.0f;
        for (float time : times)
            mean += time;
        mean /= static_cast<float>(times.size());

        printf("============================================================\n");
        printf("Mean DBFS update time: %f (success rate: %float)\n",
               mean, static_cast<float>(times.size()) / static_cast<float>(benchmarks_count));

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