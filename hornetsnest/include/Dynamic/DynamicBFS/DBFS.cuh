#pragma once

#include "HornetAlg.hpp"
#include <BufferPool.cuh>
#include <LoadBalancing/LogarithmRadixBinning.cuh>
#include <Static/BreadthFirstSearch/TopDown2.cuh>
#include <Device/Util/Timer.cuh>
#include <cuda_runtime_api.h>

namespace hornets_nest {

using vert_t = int;
using dist_t = int;

using BatchUpdate = hornet::gpu::BatchUpdate<vert_t>;

template<typename HornetGraph>
class DynamicBFS : public StaticAlgorithm<HornetGraph> {
public:

    // Contains useful statistics
    struct Stats {
        int total_frontier_expansions { 0 };
        int total_visited_vertices { 0 };
    };

public:
    DynamicBFS(HornetGraph& graph, HornetGraph& parent_graph);
    virtual ~DynamicBFS();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override;

    void set_source(vert_t source);
    dist_t get_current_level() const;
    Stats get_stats() const;
    dist_t* get_host_distance_vector() const;

    /// @brief Process the inserted or removed edges to generate an updated distance array
    void update(const vert_t* src_vertices, const vert_t* dst_vertices, int count);
    void update(BatchUpdate& batch);

    void print_nodes() const;
    void print_children(vert_t vertex);
    void print_parents(vert_t vertex);

private:
    HornetGraph& _graph;
    HornetGraph& _parent_graph;

    BufferPool _buffer_pool;
    load_balancing::BinarySearch _load_balancing;
    TwoLevelQueue<vert_t> _frontier;
    Stats _stats { };

    dist_t *_distances { nullptr };
    vert_t _source { 0 };
    dist_t _current_level { 0 };
};

} // namespace hornets_nest

namespace hornets_nest {

//static const dist_t INF = std::numeric_limits<dist_t>::max() - 1;

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//                          Algorithm Operators
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

struct Reset {
    dist_t* distances;
    OPERATOR(Vertex &vertex) {
        distances[vertex.id()] = INF;
    }
};

struct Expand {

    dist_t current_level;
    dist_t *d_distances;
    TwoLevelQueue<vert_t> queue;

    OPERATOR(Vertex &vertex, Edge &edge) {
        //printf("Expanding %d -> %d\n", vertex.id(), edge.dst_id());
        auto dst = edge.dst_id();
        if (d_distances[dst] == INF) {
            if (atomicCAS(d_distances + dst, INF, current_level) == INF) {
                queue.insert(dst);

                /*
                printf("\tAdded new node to frontier and set new level for node %d: %d\n",
                       edge.dst_id(), d_distances[dst]);
                */
            }
        }
    }
};

struct PrintNeighbours {
    vert_t target_id;
    OPERATOR(Vertex& vertex, Edge& edge) {
        if (vertex.id() == target_id)
            printf("edge: %d -> %d\n", edge.src_id(), edge.dst_id());
    }
};

// Find the smallest parent of the vertex
struct VertexUpdate {

    dist_t *distances;
    TwoLevelQueue<vert_t> frontier;

    OPERATOR(Vertex& vertex, Edge& edge) {
        //printf("Analyzing edge %d[d: %d] -> %d[d: %d] for best father\n",
        //       vertex.id(), distances[vertex.id()], edge.dst_id(), distances[edge.dst_id()]);

#if 1
        // WORKING VERSION
        const bool better_distance = distances[edge.dst_id()] < distances[vertex.id()] - 1;
        if (better_distance) {
            atomicMin(&distances[vertex.id()], distances[edge.dst_id()] + 1);
            frontier.insert(vertex.id());

            //printf("\tNew smallest father for %d[d: %d]: %d[d: %d]\n",
            //       vertex.id(), distances[vertex.id()], edge.dst_id(), distances[edge.dst_id()]);
        }
#else
        atomicMin(&distances[vertex.id()], distances[edge.dst_id()] + 1);
#endif
    }
};

// Foreach node children overwrite old distance if we are closer
struct DynamicExpandEdge {

    dist_t* distances;
    TwoLevelQueue<vert_t> frontier;

    OPERATOR(Vertex& vertex, Edge& edge) {
        //printf("Dynamic expand from %d[%d] -> %d[%d]\n",
        //       vertex.id(), distances[vertex.id()], edge.dst_id(), distances[edge.dst_id()]);

#if 1
        // WORKING VERSION

        const bool better_distance = distances[edge.dst_id()] - distances[vertex.id()] > 1;
        if (better_distance) {
            atomicMin(&distances[edge.dst_id()], distances[vertex.id()] + 1);
            frontier.insert(edge.dst_id()); // <-- TODO: Solve duplicate nodes problems

            //printf("\tNode %d has been updated to distance %d and added to frontier\n",
            //       edge.dst_id(), distances[edge.dst_id()]);
        }
#else
        const dist_t old_distance = distances[edge.dst_id()];
        if (atomicMin(&distances[edge.dst_id()], distances[vertex.id()] + 1) == old_distance)
            frontier.insert(edge.dst_id());
#endif
    }
};

//------------------------------------------------------------------------------

template<typename HornetGraph>
DynamicBFS<HornetGraph>::DynamicBFS(HornetGraph& graph, HornetGraph& parent_graph)
: StaticAlgorithm<HornetGraph>{graph}, _load_balancing{graph}, _frontier{graph}, _graph{graph}, _parent_graph{parent_graph}
{
    _buffer_pool.allocate(&_distances, _graph.nV());
    reset();
}

template<typename HornetGraph>
DynamicBFS<HornetGraph>::~DynamicBFS() { }

template<typename HornetGraph>
void DynamicBFS<HornetGraph>::set_source(vert_t source) {
    _source = source;
    _frontier.insert(_source);
    gpu::memsetZero(_distances + _source);
}

template<typename HornetGraph>
dist_t DynamicBFS<HornetGraph>::get_current_level() const {
    return _current_level;
}

template<typename HornetGraph>
auto DynamicBFS<HornetGraph>::get_stats() const -> Stats {
    return _stats;
}

template<typename HornetGraph>
dist_t* DynamicBFS<HornetGraph>::get_host_distance_vector() const {
    dist_t* result = new dist_t[_graph.nV()];
    gpu::copyToHost(_distances, _graph.nV(), result);
    return result;
}

template <typename HornetGraph> 
void DynamicBFS<HornetGraph>::reset() 
{
    _current_level = 1;
    _frontier.clear();

    dist_t* distances = _distances;
    forAllnumV(StaticAlgorithm<HornetGraph>::hornet, [=] __device__(int i) {
        distances[i] = INF;
    });
}

template<typename HornetGraph>
void DynamicBFS<HornetGraph>::print_nodes() const {
    forAllnumV(_graph, [] __device__ (int i) {
        printf("node %d\n", i);
    });
    cudaDeviceSynchronize();
}

template<typename HornetGraph>
void DynamicBFS<HornetGraph>::print_children(vert_t vertex) {
    printf("Children of node %d:\n", vertex);
    forAllEdges(_graph, PrintNeighbours { vertex }, _load_balancing);
    cudaDeviceSynchronize();
}

template<typename HornetGraph>
void DynamicBFS<HornetGraph>::print_parents(vert_t vertex) {
    printf("Parents of node %d:\n", vertex);
    forAllEdges(_parent_graph, PrintNeighbours { vertex }, _load_balancing);
    cudaDeviceSynchronize();
}

// Normal BFS
template<typename HornetGraph>
void DynamicBFS<HornetGraph>::run()
{
    assert(_frontier.size() != 0);
    while (_frontier.size() != 0) {

        forAllEdges(StaticAlgorithm<HornetGraph>::hornet,
                    _frontier,
                    Expand { _current_level, _distances, _frontier },
                    _load_balancing);

        _current_level += 1;
        _frontier.swap();
    }

#if 0
    // Print distance vector
    cudaDeviceSynchronize();
    auto* dbfs_host_distances = new dist_t[_graph.nV()];
    gpu::copyToHost(_distances, _graph.nV(), dbfs_host_distances);
    for (int i = 0; i < _graph.nV(); i++)
        printf("[DBFS] node: %d | dist: %d\n", i, dbfs_host_distances[i]);
    delete[] dbfs_host_distances;
#endif
}

template<typename HornetGraph>
void DynamicBFS<HornetGraph>::update(BatchUpdate& batch) {

    auto in_edge_soa = batch.in_edge().get_soa_ptr();
    vert_t* dst_vertices = in_edge_soa.template get<0>();
}

template<typename HornetGraph>
void DynamicBFS<HornetGraph>::update(const vert_t* src_vertices, const vert_t* dst_vertices, int count)
{
    _stats.total_frontier_expansions = 0;
    _stats.total_visited_vertices = 0;

    // Find smallest parent
    //printf("VERTEX UPDATE =============================\n");
    _frontier.insert(dst_vertices, count);
    forAllEdges(_parent_graph, _frontier, VertexUpdate { _distances, _frontier }, _load_balancing);
    _frontier.swap();

    // Propagate change to all nodes
    while (_frontier.size() > 0) {

        cudaDeviceSynchronize();
        //printf("DYNAMIC EXPANSION =========================\n");
        //printf("Expanding frontier with %d nodes\n", _frontier.size());

        _stats.total_frontier_expansions += 1;
        _stats.total_visited_vertices += _frontier.size();

        forAllEdges(_graph, _frontier, DynamicExpandEdge {  _distances, _frontier }, _load_balancing);
        _frontier.swap();
    }
}

template<typename HornetGraph>
void DynamicBFS<HornetGraph>::release() { }

template<typename HornetGraph>
bool DynamicBFS<HornetGraph>::validate() 
{
    const auto nV = StaticAlgorithm<HornetGraph>::hornet.nV();

    // Launch static BFS on the current graph
    BfsTopDown2<HornetGraph> BFS(StaticAlgorithm<HornetGraph>::hornet);
    BFS.set_parameters(_source);
    BFS.run();

    // Copy normal BFS distance vector
    auto* nbfs_host_distances = new dist_t[nV];
    gpu::copyToHost(BFS.get_distance_vector(), nV, nbfs_host_distances);
#if 0
    for (int i = 0; i < nV; i++)
        printf("[NBFS] node: %d | dist: %d\n", i, nbfs_host_distances[i]);
#endif

    // Copy dynamic BFS distance vector
    auto* dbfs_host_distances = new dist_t[nV];
    gpu::copyToHost(_distances, nV, dbfs_host_distances);
#if 0
    for (int i = 0; i < nV; i++)
        printf("[DBFS] node: %d | dist: %d\n", i, dbfs_host_distances[i]);
#endif

    // They must be the same
    int error_count = 0;
    for (int i = 0; i < nV; i++) {
        if (nbfs_host_distances[i] != INF && nbfs_host_distances[i] != dbfs_host_distances[i]) {
            error_count += 1;
            printf("[%6d] vertex %6d | nbfs: %6d, dbfs: %6d\n",
                   error_count, i, nbfs_host_distances[i], dbfs_host_distances[i]);
#if 1
            if (error_count >= 10) {
                printf("... and other errors\n");
                break;
            }
#endif
        }
    }

    delete[] nbfs_host_distances;
    delete[] dbfs_host_distances;

    return error_count == 0;
};

} // namespace hornets_nest