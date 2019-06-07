#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "MCHSimulation/Digit.h"
#include <iostream>
#endif

void loadtime() {
  o2::mch::Digit d(1.2, 42, 34.56);
  std::cout << "A" << d.getADC() << "\n";
}
