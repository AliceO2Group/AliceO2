#include <iostream>
#define O2_SIGNPOST_DEFINE_CONTEXT
#include "Framework/Signpost.h"

int main(int argc, char** argv)
{
  // To be run inside some profiler (e.g. instruments) to make sure it actually
  // works.
  O2_SIGNPOST_INIT();
  std::cout << gDPLLog << std::endl;
  std::cout << os_signpost_enabled(gDPLLog) << std::endl;
  O2_SIGNPOST(dpl, 1000, 0, 0, 0);
  O2_SIGNPOST_START(dpl, 1, 0, 0, 0);
  O2_SIGNPOST_END(dpl, 1, 0, 0, 0);
}
