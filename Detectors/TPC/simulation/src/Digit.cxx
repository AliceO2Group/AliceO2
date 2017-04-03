/// \file Digit.cxx
/// \brief Implementation of the Digit
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCSimulation/Digit.h"

ClassImp(AliceO2::TPC::Digit)

using namespace AliceO2::TPC;

Digit::Digit()
  : FairTimeStamp()
  , mMClabel(0)
  , mCharge(0.)
  , mCommonMode(-1.)
  , mCRU(-1)
  , mRow(-1)
  , mPad(-1)
{}

Digit::Digit(std::vector<long> &MClabel, int cru, float charge, int row, int pad, int time)
  : FairTimeStamp(time)
  , mMClabel(MClabel)
  , mCharge(charge)
  , mCommonMode(-1.)
  , mCRU(cru)
  , mRow(row)
  , mPad(pad)
{
}

Digit::Digit(std::vector<long> &MClabel, int cru, float charge, int row, int pad, int time, float commonMode)
  : FairTimeStamp(time)
  , mMClabel(MClabel)
  , mCharge(charge)
  , mCommonMode(commonMode)
  , mCRU(cru)
  , mRow(row)
  , mPad(pad)
{
}


std::ostream &Digit::Print(std::ostream &output) const{
  output << "TPC Digit in CRU [" << mCRU << "], pad row [" << mRow << "] and pad [" << mPad << "] with charge " << mCharge << " at time stamp" /* << mTimeStamp*/;
  return output;
}
