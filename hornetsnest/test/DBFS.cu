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

    int batch_size = std::stoi(argv[2]);
    graph::GraphStd<vert_t, vert_t> host_graph;
    host_graph.read(argv[1]);

    // Load graph into device
    HornetInit graph_init{host_graph.nV(), host_graph.nE(), host_graph.csr_out_offsets(), host_graph.csr_out_edges()};
    HornetGraph device_graph{graph_init};

    // Load inverse graph into the device
    HornetInit graph_init_inv{host_graph.nV(), host_graph.nE(), host_graph.csr_in_offsets(), host_graph.csr_in_edges()};
    HornetGraph device_graph_inv{graph_init};

    DynamicBFS<HornetGraph> DBFS{device_graph, device_graph_inv};
    DBFS.set_source(0);
    DBFS.run();

    //DBFS.print_edges();

    // =======================================================================
    // Create and apply new batch undirected

    vert_t* batch_src = new vert_t[batch_size];
    vert_t* batch_dst = new vert_t[batch_size];

    printf("Generating batch of %d edges\n", batch_size);
    generateBatch(host_graph, batch_size, batch_src, batch_dst, BatchGenType::INSERT, batch_gen_property::UNIQUE);

    // From src to dst
    HornetBatchUpdatePtr update_src_dst_ptr{batch_size, batch_src, batch_dst};
    HornetBatchUpdate update_src_dst{update_src_dst_ptr};
    device_graph.insert(update_src_dst, true, true);

    printf("===================================\n");
    printf("Batch edges:\n");
    update_src_dst.print();
    printf("===================================\n");

    // From dst to src
    HornetBatchUpdatePtr update_dst_src_ptr{batch_size, batch_dst, batch_src};
    HornetBatchUpdate update_dst_src{update_dst_src_ptr};
    device_graph_inv.insert(update_dst_src, true, true);

    // =======================================================================

    DBFS.batchUpdate(batch_dst, batch_size);

    printf("===================================\n");
    bool valid = DBFS.validate();
    printf("Validation result: %d\n", valid);

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