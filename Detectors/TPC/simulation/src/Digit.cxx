/// \file Digit.cxx
/// \author Andi Mathis, andreas.mathis@ph.tum.de

#include "TPCSimulation/Digit.h"

ClassImp(AliceO2::TPC::Digit)

using namespace AliceO2::TPC;

Digit::Digit()
  : FairTimeStamp(),
    mMCEventID(-1),
    mMCTrackID(-1),
    mCRU(-1),
    mCharge(0.),
    mRow(-1),
    mPad(-1),
    mCommonMode(-1.)
{}

Digit::Digit(Int_t eventID, Int_t trackID, Int_t cru, Float_t charge, Int_t row, Int_t pad, Int_t time)
  : FairTimeStamp(time),
    mMCEventID(eventID),
    mMCTrackID(trackID),
    mCRU(cru),
    mCharge(charge),
    mRow(row),
    mPad(pad),
    mCommonMode(-1.)
{
}

Digit::Digit(Int_t eventID, Int_t trackID, Int_t cru, Float_t charge, Int_t row, Int_t pad, Int_t time, Float_t commonMode)
  : FairTimeStamp(time),
    mMCEventID(eventID),
    mMCTrackID(trackID),
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
