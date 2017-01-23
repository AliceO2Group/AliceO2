/// \file Digit.cxx
/// \brief Digits structure for TPC digits

#include "TPCSimulation/Digit.h"

ClassImp(AliceO2::TPC::Digit)

using namespace AliceO2::TPC;

Digit::Digit()
  : FairTimeStamp(),
    mCRU(-1),
    mCharge(0.),
    mRow(-1),
    mPad(-1),
    mCommonMode(-1.)
{}

Digit::Digit(Int_t cru, Float_t charge, Int_t row, Int_t pad, Int_t time)
  : FairTimeStamp(time),
    mCRU(cru),
    mCharge(charge),
    mRow(row),
    mPad(pad),
    mCommonMode(-1.)
{
}

Digit::Digit(Int_t cru, Float_t charge, Int_t row, Int_t pad, Int_t time, Float_t commonMode)
  : FairTimeStamp(time),
    mCRU(cru),
    mCharge(charge),
    mRow(row),
    mPad(pad),
    mCommonMode(commonMode)
{
}

Digit::~Digit()
{}

std::ostream &Digit::Print(std::ostream &output) const
{
  output << "TPC Digit in CRU [" << mCRU << "], pad row [" << mRow << "] and pad [" << mPad << "] with charge " << mCharge << " at time stamp" /* << mTimeStamp*/;
  return output;
}
