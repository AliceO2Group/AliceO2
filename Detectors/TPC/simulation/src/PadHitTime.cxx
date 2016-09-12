#include "TPCSimulation/PadHitTime.h"

ClassImp(AliceO2::TPC::PadHitTime)

using namespace AliceO2::TPC;


PadHitTime::PadHitTime():
mTimeBin(),
mCharge()
{
}

PadHitTime::PadHitTime(Double_t time, Double_t charge):
mTimeBin(time),
mCharge(charge)
{
}

PadHitTime::~PadHitTime(){}
