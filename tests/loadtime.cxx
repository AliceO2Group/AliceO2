// #include "MCHSimulation/Digit.h"
// #include "TSystem.h"
#include <dlfcn.h>
#include <iostream>

class A {
public:
  A() {
    // o2::mch::Digit d(1.2, 42, 34.56);
    // std::cout << "A" << d.getADC() << "\n";
    // gSystem->Load("libMCHSimulation");
    // auto p =
    //     dlopen("/Users/laurent/alice/qc/sw/osx_x86-64/O2/0_O2_DATAFLOW-1/lib/"
    //            "libMCHSimulation.dylib",
    //            RTLD_NOW);
    // std::cout << "A" << p << "\n";
  }
};

static A a;

int main() {
  std::cout << "main\n";
  return 0;
}

