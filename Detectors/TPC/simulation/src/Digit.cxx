/// \file AliITSUpgradeDigi.cxx
/// \brief Digits structure for TPC digits

#include "TPCSimulation/Digit.h"

ClassImp(AliceO2::TPC::Digit)

using namespace AliceO2::TPC;


Digit::Digit():
FairTimeStamp(),
mCRU(-1),
mCharge(0.),
mRow(),
mPad()
{
}

Digit::Digit(Int_t cru, Double_t charge, Int_t row, Int_t pad, Double_t time):
FairTimeStamp(time),
mCRU(cru),
mCharge(charge),
mRow(row),
mPad(pad)
{
}

Digit::~Digit(){}

std::ostream &Digit::Print(std::ostream &output) const{
  output << "TPC Digit in CRU [" << mCRU << "], pad row [" << mRow << "] and pad [" << mPad << "] with charge " << mCharge << " at time stamp" /* << mTimeStamp*/;
  return output;
}
