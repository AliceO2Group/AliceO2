#include "HitCharge.h"

using namespace AliceO2::TPC;

HitCharge::HitCharge(Float_t charge) :
mCharge(charge)
{}

HitCharge::~HitCharge() {
  mCharge=0;
}