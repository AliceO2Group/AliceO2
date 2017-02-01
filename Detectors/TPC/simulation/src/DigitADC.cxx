/// \file DigitADC.cxx
/// \author Andi Mathis, andreas.mathis@ph.tum.de

#include "TPCSimulation/DigitADC.h"

using namespace AliceO2::TPC;

DigitADC::DigitADC()
  : mADC()
  , mEventID()
  , mTrackID()
{}

DigitADC::DigitADC(Int_t eventID, Int_t trackID, Float_t charge)
  : mADC(charge)
  , mEventID(eventID)
  , mTrackID(trackID)
{}

DigitADC::~DigitADC() 
{}