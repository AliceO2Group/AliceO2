/// \file DigitADC.cxx
/// \author Andi Mathis, andreas.mathis@ph.tum.de

#include "TPCSimulation/DigitADC.h"

using namespace AliceO2::TPC;

DigitADC::DigitADC()
  : mEventID(),
    mTrackID(),
    mADC()
{}

DigitADC::DigitADC(Int_t eventID, Int_t trackID, Float_t charge)
  : mEventID(eventID),
    mTrackID(trackID),
    mADC(charge)
{}

DigitADC::~DigitADC() = default;
