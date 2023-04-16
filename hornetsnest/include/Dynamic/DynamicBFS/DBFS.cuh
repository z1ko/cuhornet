#pragma once

#include "HornetAlg.hpp"
#include <BufferPool.cuh>
#include <LoadBalancing/LogarithmRadixBinning.cuh>
#include <Static/BreadthFirstSearch/TopDown2.cuh>
#include <Device/Util/Timer.cuh>

#include <stdio.h>

namespace hornets_nest {

using vert_t = int;
using dist_t = int;

using BatchUpdate = hornet::gpu::BatchUpdate<vert_t>;

template<typename HornetGraph>
class DynamicBFS : public StaticAlgorithm<HornetGraph> {
public:
    DynamicBFS(HornetGraph& graph, HornetGraph& parent_graph);
    virtual ~DynamicBFS();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override;

    void set_source(vert_t source);
    dist_t get_current_level() const;

    /// @brief Process the inserted or removed edges to generate an updated distance array
    void batchUpdate(const vert_t* dst_vertices, int count);
    void batchUpdate(BatchUpdate& batch);

    void print_edges() const;

private:
    HornetGraph& _parent_graph;

    BufferPool _buffer_pool;
    load_balancing::LogarthimRadixBinning32 _load_balancing;
    TwoLevelQueue<vert_t> _frontier;

    dist_t* _distances { nullptr };
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
    dist_t *distances;

    TwoLevelQueue<vert_t> frontier;
  
    OPERATOR(Vertex &vertex, Edge &edge) {
        auto dst = edge.dst_id();

        /*
        printf("src: %d, dst: %d, src_distance: %d, dst_distance: %d\n", 
            vertex.id(), dst, distances[vertex.id()], distances[dst]);
        */

        if (distances[dst] == INF) {
            if (atomicCAS(distances + dst, INF, current_level) == INF)
                frontier.insert(dst);
        }
    }
};

struct Printer {
    dist_t* distances;
    OPERATOR(Vertex& src, Edge& edge) {
        printf("src: %d, dst: %d, src_distance: %d, dst_distance: %d\n", 
            src.id(), edge.dst_id(), distances[src.id()], distances[edge.dst_id()]);
    }
};

struct FindSmallestDistance {

    dist_t* distances;

    OPERATOR(Vertex& src, Edge& edge) {

        auto src_id = src.id();
        auto dst_id = edge.dst_id();

        atomicMin(distances + src_id, distances[dst_id] + 1);
    }
};

struct DynamicExpand {

    TwoLevelQueue<vert_t> frontier;
    dist_t* distances;

    OPERATOR(Vertex& src, Edge& edge) {
        
        auto src_id = src.id();
        auto dst_id = edge.dst_id();

        /*
         * if (atomicMin(&distances[dst_id], distances[src_id] + 1) == dst_distance)
            frontier.insert(dst_id);
         */

        // If the destination has a bigger distance than us then it must be updated
        const auto dst_distance = distances[dst_id];
        if (dst_distance > distances[src_id] + 1) {
            // Multiple nodes could be trying to update this node distance
            if (atomicMin(distances + dst_id, distances[src_id] + 1) == dst_distance)
                frontier.insert(dst_id);
        }
    }
};

struct PrintEdge {
    OPERATOR(Vertex& src, Edge& e) {
        printf("%4d -> %4d\n", e.src_id(), e.dst_id());
    }
};

/*
struct InitializeUpdatedVertices {

    dist_t *distances;

    OPERATOR(vert_t v) {

        auto src_id = src.id();
        auto dst_id = dst.id();

        if (distances[dst_id] + 1 < distances[src_id]) {
            atomicMin(distances + src_id, distances[dst_id] + 1);
        }
    }
};
*/

//------------------------------------------------------------------------------

template<typename HornetGraph>
DynamicBFS<HornetGraph>::DynamicBFS(HornetGraph& graph, HornetGraph& parent_graph)
: StaticAlgorithm<HornetGraph>{graph}, _load_balancing{graph}, _frontier{graph}, _parent_graph{graph}
{
    _buffer_pool.allocate(&_distances, graph.nV());
    reset();
}

template<typename HornetGraph>
DynamicBFS<HornetGraph>::~DynamicBFS() { }

template<typename HornetGraph>
void DynamicBFS<HornetGraph>::set_source(vert_t source) { _source = source; }

template<typename HornetGraph>
dist_t DynamicBFS<HornetGraph>::get_current_level() const {
    return _current_level;
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

    gpu::memsetZero(_distances + _source);
}

template<typename HornetGraph>
void DynamicBFS<HornetGraph>::print_edges() const {
    forAllEdges(StaticAlgorithm<HornetGraph>::hornet, PrintEdge { }, _load_balancing);
}

template<typename HornetGraph>
void DynamicBFS<HornetGraph>::run()
{
    // Normal bfs
    _frontier.insert(_source);
    while (_frontier.size() > 0) {

        forAllEdges(
            StaticAlgorithm<HornetGraph>::hornet, 
            _frontier,
            Expand { _current_level, _distances, _frontier }, 
            _load_balancing
        );

        _frontier.swap();
        _current_level++;
    }
}

template<typename HornetGraph>
void DynamicBFS<HornetGraph>::batchUpdate(BatchUpdate& batch) {

    auto in_edge_soa = batch.in_edge().get_soa_ptr();
    vert_t* dst_vertices = in_edge_soa.template get<0>();
    batchUpdate(dst_vertices, batch.in_edge().get_num_items());
}

template<typename HornetGraph>
void DynamicBFS<HornetGraph>::batchUpdate(const vert_t* dst_vertices, int count)
{
    /**
     * Steps of the algorithm:
     * 1) For each updated edge destination find the nearest "smallest distance from source" parent
     * 2) Add the node to the frontier
     * 3) While the frontier is not empty update using a normal BFS replacing higher than current level distances
    */

    // Add all modified nodes to the frontier to be processed
    // TODO: immediately calculate next value instead of adding to queue
    /*
    forAllEdgesBatch(StaticAlgorithm<HornetGraph>::hornet,
                batch_update,
                Printer { },
                _load_balancing);
    */

    // Add all vertices to frontier to be processed

    timer::Timer<timer::DEVICE> TM;
    TM.start();

    //printf("queue size: %d\n", _frontier.size());
    _frontier.insert(dst_vertices, count);
    //printf("queue size: %d\n", _frontier.size());

    // Reset distance from origin for all updated nodes
    //forAllVertices(StaticAlgorithm<HornetGraph>::hornet, _frontier, Reset { _distances });

    // Find smallest parent
    //forAllEdges(_parent_graph, _frontier, Printer { _distances }, _load_balancing);
    forAllEdges(_parent_graph, _frontier, FindSmallestDistance { _distances }, _load_balancing);
    //forAllEdges(_parent_graph, _frontier, Printer { _distances }, _load_balancing);

    // Propagate change to all nodes
    while (_frontier.size() > 0) {
        printf("Executing DBFS with %d vertices\n", _frontier.size());
        forAllEdges(StaticAlgorithm<HornetGraph>::hornet, _frontier, DynamicExpand { _frontier, _distances }, _load_balancing);
        _frontier.swap();
    }

    TM.stop();
    printf("DBFS Update time: %f\n", TM.duration());
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

    timer::Timer<timer::DEVICE> TM;
    TM.start();

    BFS.run();

    TM.stop();
    printf("BFS time: %f\n", TM.duration());

    // Copy normal BFS distance vector
    auto* nbfs_host_distances = new dist_t[nV];
    gpu::copyToHost(BFS.get_distance_vector(), nV, nbfs_host_distances);

    // Copy dynamic BFS distance vector
    auto* dbfs_host_distances = new dist_t[nV];
    gpu::copyToHost(_distances, nV, dbfs_host_distances);

    // They must be the same
    int error_count = 0;
    for (int i = 0; i < nV; i++) {
        if (nbfs_host_distances[i] != dbfs_host_distances[i]) {
            error_count += 1;
            /*
            printf("[%6d] vertex %6d | nbfs: 6d, dbfs: 6d\n",
                   error_count, i, nbfs_host_distances[i], dbfs_host_distances[i]);
            */
        }
    }

    delete[] nbfs_host_distances;
    delete[] dbfs_host_distances;

    return error_count == 0;
};

} // namespace hornets_nest