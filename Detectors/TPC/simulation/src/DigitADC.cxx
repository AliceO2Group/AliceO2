/// \file DigitADC.cxx
/// \author Andi Mathis, andreas.mathis@ph.tum.de

#include "TPCSimulation/DigitADC.h"

using namespace AliceO2::TPC;

DigitADC::DigitADC()
  : mADC()
  , mEventID()
  , mTrackID()
{}

DigitADC::DigitADC(int eventID, int trackID, float charge)
  : mADC(charge)
  , mEventID(eventID)
  , mTrackID(trackID)
{}

DigitADC::~DigitADC() 
{}