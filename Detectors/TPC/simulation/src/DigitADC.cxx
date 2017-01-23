#include "TPCSimulation/DigitADC.h"

using namespace AliceO2::TPC;

DigitADC::DigitADC()
  : mADC()
{}

DigitADC::DigitADC(Float_t charge)
  : mADC(charge)
{}

DigitADC::~DigitADC() = default;