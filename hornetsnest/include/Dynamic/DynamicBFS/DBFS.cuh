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
    struct Stats 
    {
        // General frontier data
        int frontier_expansions_count;
        int initial_frontier_size;

        // Time measures for dynamic update
        float vertex_update_time;
        float expansion_time;

        float bfs_time;
        int bfs_max_level;
    };

public:
    DynamicBFS(HornetGraph& graph, HornetGraph& parent_graph, int max_update_size);
    virtual ~DynamicBFS();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override;

    void set_source(vert_t source);
    dist_t get_current_level() const;
    Stats get_stats() const;
    dist_t* get_host_distance_vector() const;

    void set_device_distance_vector(int* distances);

    /// @brief Process the inserted or removed edges to generate an updated distance array
    void update(const vert_t* dst_vertices, int count);
    void update(BatchUpdate& batch);

    void print_nodes() const;
    void print_children(vert_t vertex);
    void print_parents(vert_t vertex);

    /// @brief Return current distance vector
    dist_t* current_distances();
    /// @brief Return writable distance vector
    dist_t* write_distances();

    /// @brief Swap the current distance vector
    void swap_distances();
    /// #brief Copy the current distances into the others
    void sync_distances();

private:
    HornetGraph& _graph;
    HornetGraph& _parent_graph;

    BufferPool _buffer_pool;
    load_balancing::BinarySearch _load_balancing;
    TwoLevelQueue<vert_t> _frontier;
    Stats _stats { };

    // Used to filter only the usefull edges from the batch
    int _max_update_size { 0 };
    int *_valid_edges { nullptr };

    int _current_distance_vector { 0 };
    dist_t *_distances[2] = { nullptr, nullptr };

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

    dist_t *read_distances;
    dist_t *write_distances;

    TwoLevelQueue<vert_t> frontier;

    OPERATOR(Vertex& vertex, Edge& edge) {
        //printf("Analyzing edge %d[d: %d] -> %d[d: %d] for best father\n",
        //       vertex.id(), read_distances[vertex.id()], edge.dst_id(), read_distances[edge.dst_id()]);

#if 1
        // WORKING VERSION
        
        const int distance = read_distances[vertex.id()];
        if (read_distances[edge.dst_id()] < distance - 1) {
            if (atomicMin(&write_distances[vertex.id()], read_distances[edge.dst_id()] + 1) == distance) {
              frontier.insert(vertex.id());
            }

            //printf("\tNew smallest father for %d: %d at distance %d\n",
            //       vertex.id(), edge.dst_id(), read_distances[edge.dst_id()]);
        }
#else
        atomicMin(&distances[vertex.id()], distances[edge.dst_id()] + 1);
#endif
    }
};

// Foreach node children overwrite old distance if we are closer
struct DynamicExpandEdge {

    dist_t* read_distances;
    dist_t* write_distances;

    TwoLevelQueue<vert_t> frontier;

    OPERATOR(Vertex& vertex, Edge& edge) {
        //printf("Dynamic expand from %d[%d] -> %d[%d]\n",
        //       vertex.id(), read_distances[vertex.id()], edge.dst_id(), read_distances[edge.dst_id()]);

#if 1
        // WORKING VERSION

        const int our_dist = read_distances[vertex.id()];
        const int dst_dist = read_distances[edge.dst_id()];

        const int delta = dst_dist - our_dist;
        if (delta > 1) {

            // Modify distance of destination node, if the operation returns our read value
            // then we must add to queue
            if (atomicMin(&write_distances[edge.dst_id()], our_dist + 1) == dst_dist) {
              frontier.insert(edge.dst_id());
            }

            //printf("\tNode %d has been updated to distance %d\n",
            //       edge.dst_id(), our_dist + 1);
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
DynamicBFS<HornetGraph>::DynamicBFS(HornetGraph& graph, HornetGraph& parent_graph, int max_update_size)
: StaticAlgorithm<HornetGraph>{graph}, _load_balancing{graph}, _frontier{graph, 10.0f}, 
  _graph{graph}, _parent_graph{parent_graph}, _max_update_size{max_update_size}
{
    _buffer_pool.allocate(&_distances[0], _graph.nV());
    _buffer_pool.allocate(&_distances[1], _graph.nV());

    _buffer_pool.allocate(&_valid_edges, _max_update_size);
    reset();
}

template<typename HornetGraph>
DynamicBFS<HornetGraph>::~DynamicBFS() { }

template<typename HornetGraph>
void DynamicBFS<HornetGraph>::set_source(vert_t source) {
    _source = source;
}

template<typename HornetGraph>
dist_t DynamicBFS<HornetGraph>::get_current_level() const {
    return _current_level;
}

template<typename HornetGraph>
void DynamicBFS<HornetGraph>::set_device_distance_vector(int* distances) {
  cudaMemcpy(current_distances(), distances, _graph.nV(), cudaMemcpyDeviceToDevice); 
}

template<typename HornetGraph>
auto DynamicBFS<HornetGraph>::get_stats() const -> Stats {
    return _stats;
}

template<typename HornetGraph>
dist_t* DynamicBFS<HornetGraph>::get_host_distance_vector() const {
    dist_t* result = new dist_t[_graph.nV()];
    gpu::copyToHost(_distances[_current_distance_vector], _graph.nV(), result);
    return result;
}

template<typename HornetGraph>
dist_t* DynamicBFS<HornetGraph>::current_distances() {
  return _distances[_current_distance_vector];
}

template<typename HornetGraph>
dist_t* DynamicBFS<HornetGraph>::write_distances() {
  return _distances[!_current_distance_vector];
}

template<typename HornetGraph>
void DynamicBFS<HornetGraph>::swap_distances() {
  _current_distance_vector = !_current_distance_vector;
}

// Copy the content of the 'from' vector to the 'to' vector
__global__ void sync_distances_kernel(const dist_t* from, dist_t* to, int size) {

    int stride = blockDim.x * gridDim.x;
    int  start = blockIdx.x * blockDim.x + threadIdx.x;

    for (auto i = start; i < size; i += stride) {
        to[i] = from[i]; 
    }
}

template<typename HornetGraph>
void DynamicBFS<HornetGraph>::sync_distances() {
  dist_t* src_distances = _distances[_current_distance_vector];
  dist_t* dst_distances = _distances[!_current_distance_vector];

  int block_count = xlib::ceil_div<1024>(_graph.nV());
  sync_distances_kernel<<<block_count, 1024>>>(src_distances, dst_distances, _graph.nV());
}

template <typename HornetGraph> 
void DynamicBFS<HornetGraph>::reset() 
{
    _current_level = 1;
    _frontier.clear();

    dist_t* distances = current_distances();
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
    _frontier.insert(_source);
    gpu::memsetZero(current_distances() + _source);
    
    assert(_frontier.size() != 0);
    while (_frontier.size() != 0) {

        forAllEdges(StaticAlgorithm<HornetGraph>::hornet,
                    _frontier,
                    Expand { _current_level, current_distances(), _frontier },
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

struct FilterEdges {

  int *valid;
  dist_t* distances;
  
  OPERATOR(Vertex& src, Vertex& dst) {
    const int delta = distances[dst.id()] - distances[src.id()];
    printf("Batch edge: %d -> %d, delta: %d\n", 
        src.id(), dst.id(), delta);

    // Write result
  }
};

template<typename HornetGraph>
void DynamicBFS<HornetGraph>::update(BatchUpdate& batch) {

    const int batch_size = batch.size();
    assert(batch_size <= _max_update_size);

    forAllEdgesBatch(_graph, batch, FilterEdges { _valid_edges, current_distances() });

    auto batch_soa_ptr = batch.in_edge().get_soa_ptr();
    update(batch_soa_ptr.template get<1>(), batch_size); 
}

#if 0
  #define CSV_DBFS_OUTPUT
#endif

  template<typename HornetGraph>
void DynamicBFS<HornetGraph>::update(const vert_t* dst_vertices, int count)
{
    // Sync the distance vectors
    sync_distances();

    // Used to benchmark all code parts
    timer::Timer<timer::DEVICE> section_timer;

#ifdef CSV_DBFS_OUTPUT 
    std::cerr << "seq\tfrontier_size" << std::endl;
    int index = 0;
#endif

    section_timer.start();

    // Find smallest parent
    _frontier.insert(dst_vertices, count);
    forAllEdges(_parent_graph, _frontier, 
        VertexUpdate { current_distances(), write_distances(), _frontier }, 
        _load_balancing);

    swap_distances();
    sync_distances();
    _frontier.swap();

    section_timer.stop();
    
    _stats.vertex_update_time = section_timer.duration();
    //section_timer.print("Vertex Update");

    section_timer.start();

    // Propagate change to all nodes
    _stats.initial_frontier_size = _frontier.size();
    while (_frontier.size() > 0) {
        cudaDeviceSynchronize(); 

        const int frontier_size = _frontier.size();
        _stats.frontier_expansions_count += 1;

#ifdef CSV_DBFS_OUTPUT
        std::cerr << index << "\t" << frontier_size << std::endl;
        index += 1;
#endif

        // Propagate new distances
        forAllEdges(_graph, _frontier, 
            DynamicExpandEdge { current_distances(), write_distances(), _frontier }, 
            _load_balancing);

        swap_distances();
        sync_distances();
        
        _frontier.swap();
    }
    
    section_timer.stop();

    _stats.expansion_time = section_timer.duration();
    //section_timer.print("Dynamic expand");
}

template<typename HornetGraph>
void DynamicBFS<HornetGraph>::release() {
    gpu::free(_distances[0], _graph.nV());
    gpu::free(_distances[1], _graph.nV());
    gpu::free(_valid_edges, _max_update_size);
}

template<typename HornetGraph>
bool DynamicBFS<HornetGraph>::validate() 
{
    const auto nV = StaticAlgorithm<HornetGraph>::hornet.nV();
    timer::Timer<timer::DEVICE> timer;

    // Launch static BFS on the current graph
    BfsTopDown2<HornetGraph> BFS(_graph);
    BFS.set_parameters(_source);

    timer.start();
    BFS.run();
    timer.stop();

    _stats.bfs_time = timer.duration();
    _stats.bfs_max_level = BFS.getLevels();

    // Copy normal BFS distance vector
    auto* nbfs_host_distances = new dist_t[nV];
    gpu::copyToHost(BFS.get_distance_vector(), nV, nbfs_host_distances);

#if 0
    for (int i = 0; i < nV; i++)
        printf("[NBFS] node: %d | dist: %d\n", i, nbfs_host_distances[i]);
#endif

    // Copy dynamic BFS distance vector
    auto* dbfs_host_distances = new dist_t[nV];
    gpu::copyToHost(current_distances(), nV, dbfs_host_distances);

#if 0
    for (int i = 0; i < nV; i++)
        printf("[DBFS] node: %d | dist: %d\n", i, dbfs_host_distances[i]);
#endif

    // They must be the same
    int error_count = 0;
    for (int i = 0; i < nV; i++) {
        
        // If the nbfs distance is INF then the node is not in reachable from source
        if (nbfs_host_distances[i] != dbfs_host_distances[i]) {
            error_count += 1;
            printf("[%6d] vertex %6d | nbfs: %6d, dbfs: %6d\n",
                   error_count, i, nbfs_host_distances[i], dbfs_host_distances[i]);
#if 0
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
