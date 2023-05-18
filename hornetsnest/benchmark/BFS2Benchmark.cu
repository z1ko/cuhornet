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

        timer::Timer<timer::DEVICE> TM;
        std::vector<float> measures;
  
        cudaEvent_t start, stop;

        int benchmarks_count = std::stoi(argv[2]);
        for (int benchmark = 0; benchmark < benchmarks_count; benchmark++) {
            
            BfsTopDown2<HornetGraph> BFS(device_graph);
            BFS.set_parameters(device_graph.max_degree_id());
            
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            BFS.run();
            cudaEventRecord(stop);

            cudaEventSynchronize(stop);
            float duration;
            cudaEventElapsedTime(&duration, start, stop);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            measures.push_back(duration);
            std::cout << "elapsed: " << duration << "\n";

            BFS.release();
        }

        float elapsed_mean = 0.0f;
        for (float measure : measures)
          elapsed_mean += measure;
        elapsed_mean /= static_cast<float>(measures.size());

        std::cout << "BFS mean time: " << elapsed_mean << "\n";
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
