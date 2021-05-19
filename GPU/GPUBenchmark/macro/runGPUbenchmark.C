#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <GPUBenchmark/GPUbenchmark.h>
#endif

void runGPUbenchmark()
{
  o2::benchmark::GPUbenchmark bm{};
  bm.hello();
}