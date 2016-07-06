/// \file AliITSUpgradeDigi.cxx
/// \brief Digits structure for ITS digits

#include "Digit.h"

ClassImp(AliceO2::TPC::Digit)

using namespace AliceO2::TPC;


Digit::Digit():
mCRU(-1),
mCharge(0.),
mRow(),
mPad(),
FairTimeStamp()
{
}

Digit::Digit(Int_t cru, Double_t charge, Int_t row, Int_t pad, Double_t time):
mCRU(cru),
mCharge(charge),
mRow(row),
mPad(pad),
FairTimeStamp(time)
{
}

Digit::~Digit(){}

std::ostream &Digit::Print(std::ostream &output) const{
  output << "TPC Digit in CRU [" << mCRU << "], pad row [" << mRow << "] and pad [" << mPad << "] with charge " << mCharge << " at time stamp" /* << mTimeStamp*/;
  return output;
}
