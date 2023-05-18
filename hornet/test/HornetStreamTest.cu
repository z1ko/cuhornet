#include <Hornet.hpp>
#include "StandardAPI.hpp"
#include "Util/BatchFunctions.hpp"
#include "Util/RandomGraphData.cuh"
#include <Host/FileUtil.hpp>            //xlib::extract_filepath_noextension
#include <Device/Util/CudaUtil.cuh>     //xlib::deviceInfo
#include <algorithm>                    //std:.generate
#include <chrono>                       //std::chrono
#include <random>                       //std::mt19937_64
#include <cuda_profiler_api.h>
#include <Core/Static/Static.cuh>

#include <Graph/GraphStd.hpp>
#include <Host/Classes/Timer.hpp>
#include <Device/Util/Timer.cuh>
#include "Util/CommandLineParam.hpp"

using namespace timer;
using namespace std::string_literals;
using namespace hornets_nest;

using vert_t = int;
using eoff_t = int;
using HornetGPU = hornet::gpu::Hornet<vert_t>;
using UpdatePtr = hornet::BatchUpdatePtr<vert_t, hornet::EMPTY, hornet::DeviceType::HOST>;
using Update = hornet::gpu::BatchUpdate<vert_t>;
using Init = hornet::HornetInit<vert_t>;
using hornet::SoAData;
using hornet::TypeList;
using hornet::DeviceType;

/**
 * @brief This test inserts and erases a number of batches
 */
int exec(int argc, char** argv) {
  
  if (argc != 4) {
    std::cout << "args: <graph filename> <batch size> <iterations count>\n";
    return -1;
  }
  
  int batch_size = std::stoi(argv[2]);
  int iterations = std::stoi(argv[3]);

  graph::GraphStd<vert_t, vert_t> loader;
  loader.read(argv[1]);

  Init initializer{loader.nV(), loader.nE(), loader.csr_out_offsets(), loader.csr_out_edges()};
  HornetGPU hornet{initializer};

  // Initial state of the graph
  auto initial_coo = hornet.getCOO();
  initial_coo.sort();
  
  hornet::COO<DeviceType::HOST, vert_t, hornet::EMPTY, eoff_t> host_initial_coo = initial_coo;

#if 0
  std::cout << "Initial Graph:\n";
  hornet.print();
#endif

  vert_t* src = new vert_t[batch_size];
  vert_t* dst = new vert_t[batch_size];
  
  for(int i = 0; i < iterations; i++) {
    
    std::cout << "Generating batch of " << batch_size << " elements... ";
    generateBatch(loader, batch_size, src, dst, BatchGenType::INSERT);
    std::cout << "DONE\n";

    Update update_insert{UpdatePtr{batch_size, src, dst}};
    Update update_erase{UpdatePtr{batch_size, src, dst}};
    
    std::cout << "Inserting... ";
    hornet.insert(update_insert);
    std::cout << "DONE\n";

    std::cout << "Erasing... ";
    hornet.erase(update_erase);
    std::cout << "DONE\n";
  }

  delete[] src;
  delete[] dst;

  // Final state of the graph
  auto final_coo = hornet.getCOO();
  final_coo.sort();

  hornet::COO<DeviceType::HOST, vert_t, hornet::EMPTY, eoff_t> host_final_coo = final_coo;

#if 0
  std::cout << "Final Graph:\n";
  hornet.print();
#endif

  // ======================================================================================
  // Check consistency of graph

  auto *s = host_initial_coo.srcPtr();
  auto *d = host_initial_coo.dstPtr();
    
  auto *S = host_final_coo.srcPtr();
  auto *D = host_final_coo.dstPtr();
    
  auto len = host_initial_coo.size();
  std::cout << "Edge count: " << len << std::endl;

  if (host_final_coo.size() != host_initial_coo.size()) {
    std::cout << "\nInit Size "<< host_initial_coo.size() << " != Combined size " << host_final_coo.size() << "\n";
    return -1;
  }

  bool valid = true;
  int counter = 0;
  for (int i = 0; i < len; i++) {
    bool error = (s[i] != S[i]) || (d[i] != D[i]);
    if (error) {
      valid = false;
      printf("%5d -> %5d | %5d -> %5d => %s\n",
        s[i], d[i], S[i], D[i], "ERROR");
    }
    counter += 1;
  }

  std::cout << "Effective size: " << counter << std::endl;
  std::cout << (valid ? "PASSED" : "NOT PASSED") 
            << std::endl;

  return 0;
}

int main(int argc, char* argv[]) {
  int ret = 0;
  {
    ret = exec(argc, argv);
  }

  return ret;
}
