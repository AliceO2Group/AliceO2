#include "DigitADC.h"
#include "Digit.h"   // for Digit

using namespace AliceO2::TPC;

DigitADC::DigitADC(Float_t charge) :
mADC(charge)
{}

DigitADC::~DigitADC() {}