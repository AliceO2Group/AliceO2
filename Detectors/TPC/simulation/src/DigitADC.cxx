#include "TPCsimulation/DigitADC.h"
#include "TPCsimulation/Digit.h"   // for Digit

using namespace AliceO2::TPC;

DigitADC::DigitADC(Float_t charge) :
mADC(charge)
{}

DigitADC::~DigitADC() {}
