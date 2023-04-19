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
using HornetBatchUpdatePtr = hornet::BatchUpdatePtr<vert_t, hornet::EMPTY, hornet::DeviceType::DEVICE>;
using HornetBatchUpdate = hornet::gpu::BatchUpdate<vert_t>;

// Generate a batch that requires an expansion
void generateEvilBatch(vert_t* src, vert_t* dst, int batch_size, dist_t* dist, int dist_size, int delta) {
    srand(time(0));
    for(int i = 0; i < batch_size; i++) {

        vert_t src_id = rand() % dist_size;
        vert_t dst_id = rand() % dist_size;

        while(dist[dst_id] == INF || dist[dst_id] - dist[src_id] < delta) {
            src_id = rand() % dist_size;
            dst_id = rand() % dist_size;
        }

        src[i] = src_id;
        dst[i] = dst_id;
    }
}

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

    vert_t *dev_batch_src, *dev_batch_dst;
    cudaMalloc(&dev_batch_src, sizeof(vert_t) * batch_size);
    cudaMalloc(&dev_batch_dst, sizeof(vert_t) * batch_size);

    printf("Generating batch of %d edges\n", batch_size);
#if 1
    generateBatch(host_graph, batch_size, batch_src, batch_dst, BatchGenType::INSERT, batch_gen_property::UNIQUE);
#else
    dist_t* distances = DBFS.get_host_distance_vector();
    generateEvilBatch(batch_src, batch_dst, batch_size, distances, device_graph.nV(), 2);
    delete[] distances;
#endif

    printf("Generated edges: \n");
    for(int i = 0; i < batch_size; i++) {
        printf("\t%d -> %d\n", batch_src[i], batch_dst[i]);
    }

    // Copy to device
    cudaMemcpy(dev_batch_src, batch_src, sizeof(vert_t) * batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_batch_dst, batch_dst, sizeof(vert_t) * batch_size, cudaMemcpyHostToDevice);

    // From src to dst
    HornetBatchUpdatePtr update_src_dst_ptrs{batch_size, dev_batch_src, dev_batch_dst};
    HornetBatchUpdate update_src_dst{update_src_dst_ptrs};

#if 1
    printf("=====================================\n");
    printf("Graph before and after batch         \n");

    device_graph.print();
    device_graph.insert(update_src_dst, false, false);
    printf(" --- \n");
    device_graph.print();

    // From dst to src
    HornetBatchUpdatePtr update_dst_src_ptrs{batch_size, dev_batch_dst, dev_batch_src};
    HornetBatchUpdate update_dst_src{update_dst_src_ptrs};

    printf("=====================================\n");
    printf("Inverse graph before and after batch \n");

    device_graph.print();
    device_graph_inv.insert(update_dst_src, false, false);
    printf(" --- \n");
    device_graph_inv.print();
#endif

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