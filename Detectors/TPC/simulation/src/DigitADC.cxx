#include "TPCSimulation/DigitADC.h"
#include "TPCSimulation/Digit.h"   // for Digit

using namespace AliceO2::TPC;

DigitADC::DigitADC() :
mADC()
{}

DigitADC::DigitADC(Float_t charge) :
mADC(charge)
{}

DigitADC::~DigitADC() = default;