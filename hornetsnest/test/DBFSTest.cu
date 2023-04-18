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
    host_graph.read(argv[1], PRINT_INFO | RM_SINGLETON);

    HornetInit graph_init{host_graph.nV(),
                          host_graph.nE(),
                          host_graph.csr_out_offsets(),
                          host_graph.csr_out_edges()};

    HornetInit graph_init_inv{host_graph.nV(),
                              host_graph.nE(),
                              host_graph.csr_in_offsets(),
                              host_graph.csr_in_edges()};

    HornetGraph device_graph{graph_init};
    // TODO: Inverse graph is not updated correctly, for now use undirected graph
    HornetGraph device_graph_inv{graph_init_inv};
    DynamicBFS<HornetGraph> DBFS{device_graph, device_graph_inv};

    vert_t source = device_graph.max_degree_id();
    if (argc == 4)
        source = std::stoi(argv[3]);

    DBFS.set_source(source);
    DBFS.run();

    // =======================================================================
    // Create and apply new batch undirected

    vert_t* batch_src = new vert_t[batch_size];
    vert_t* batch_dst = new vert_t[batch_size];

    printf("Generating batch of %d edges\n", batch_size);
    generateBatch(host_graph, batch_size, batch_src, batch_dst, BatchGenType::INSERT, batch_gen_property::UNIQUE);

    printf("Batch edges:\n");
    for(int i = 0; i < batch_size; i++) {
        printf("\t%d -> %d\n", batch_src[i], batch_dst[i]);
    }

    // From src to dst
    HornetBatchUpdatePtr update_src_dst_ptr{batch_size, batch_src, batch_dst};
    HornetBatchUpdate update_src_dst{update_src_dst_ptr};
    device_graph.insert(update_src_dst, false, false);

    // From dst to src
    HornetBatchUpdatePtr update_dst_src_ptr{batch_size, batch_dst, batch_src};
    HornetBatchUpdate update_dst_src{update_dst_src_ptr};
    device_graph_inv.insert(update_dst_src, false, false);

    //device_graph.insert(update_dst_src, false, false); // NOTE: Assume undirected graph



#if 0
    printf("===================================\n");
    printf("Normal graph:\n");
    device_graph.print();

    printf("===================================\n");
    printf("Inverse graph:\n");
    device_graph_inv.print();
#endif

    // =======================================================================
    // Sanity check

    auto device_graph_COO = device_graph.getCOO();
    auto device_graph_map = getHostMMap(device_graph_COO);

    auto device_graph_inv_COO = device_graph_inv.getCOO();
    auto device_graph_inv_map = getHostMMap(device_graph_inv_COO);

    // Check that all edges have been applied
    for (int i = 0; i < batch_size; i++) {

        // SRC -> DST
        bool found_child = false;
        for (auto it = device_graph_map.find(batch_src[i]); it != device_graph_map.end(); it++) {
            if (std::get<0>(it->second) == batch_dst[i]) {
                found_child = true;
                break;
            }
        }

        if (!found_child) {
            printf("[%4d] Edge %d -> %d not inserted correctly: missing child node\n",
                   i, batch_src[i], batch_dst[i]);
            DBFS.print_children(batch_src[i]);
        }

        // DST -> SRC
        bool found_parent = false;
        for (auto it = device_graph_inv_map.find(batch_dst[i]); it != device_graph_inv_map.end(); it++) {
            if (std::get<0>(it->second) == batch_src[i]) {
                found_parent = true;
                break;
            }
        }

        if (!found_parent) {
            printf("[%4d] Edge %d -> %d not inserted correctly: missing parent node\n",
                   i, batch_src[i], batch_dst[i]);
            DBFS.print_parents(batch_dst[i]);
        }
    }

    // =======================================================================

    timer::Timer<timer::DEVICE> TM;
    TM.start();
    DBFS.batchUpdate(batch_src, batch_dst, batch_size);
    TM.stop();

    auto stats = DBFS.get_stats();

    printf("===================================\n");
    printf("Update time: %f\n", TM.duration());
    printf("Total expanded frontiers: %d\n", stats.total_frontier_expansions);
    printf("Total visited vertices: %d\n", stats.total_visited_vertices);

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