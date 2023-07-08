
// Usefull for grid strided work
#define KERNEL_FOR_STRIDED(var, N)                                             \
  for (var = blockDim.x * blockIdx.x + threadIdx.x; var < N;                   \
       var += blockDim.x * gridDim.x)

// Usefull for standard thread/work usage
#define BLOCK_COUNT(size, work) ((size + work - 1) / work)

// Usefull with timers
#define INSTRUMENT(timer, result, expr)                                        \
  {                                                                            \
    timer.start();                                                             \
    expr;                                                                      \
    timer.stop();                                                              \
    result = timer.duration();                                                 \
  }
