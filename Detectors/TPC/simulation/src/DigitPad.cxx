/// \file DigitPad.cxx
/// \author Andi Mathis, andreas.mathis@ph.tum.de

#include "TPCSimulation/DigitPad.h"
#include "TPCSimulation/SAMPAProcessing.h"
#include "TPCSimulation/Digit.h"

using namespace AliceO2::TPC;

DigitPad::DigitPad(int pad)
  : mTotalChargePad(0.)
  , mPad(pad)
  , mADCCounts()
{}

DigitPad::~DigitPad()
{
  mADCCounts.resize(0);
  mTotalChargePad = 0;
}

void DigitPad::fillOutputContainer(TClonesArray *output, int cru, int timeBin, int row, int pad)
{  
  float totalADC = 0;
  for(auto &aADCCounts : mADCCounts) {
    totalADC += aADCCounts.getADC();
  }
  
  const float mADC = SAMPAProcessing::getADCSaturation(totalADC);
  
  if(mADC > 0) {
    TClonesArray &clref = *output;
    new(clref[clref.GetEntriesFast()]) Digit(-1, -1, cru, mADC, row, pad, timeBin);
  }
}

void DigitPad::fillOutputContainer(TClonesArray *output, int cru, int timeBin, int row, int pad, float commonMode)
{
  float totalADC = 0;
  int MCEventIDOld = -1;
  int MCTrackIDOld = -1;
  for(auto &aADCCounts : mADCCounts) {
    totalADC += aADCCounts.getADC();
//     int currentMCEvent = aADCCounts.getMCEventID();
//     int currentMCTrack = aADCCounts.getMCTrackID();
//     if(MCEventIDOld != currentMCEvent) {
//       MCEventIDOld = currentMCEvent;
//     }
//     if(MCTrackIDOld != currentMCTrack) {
//       MCTrackIDOld = currentMCTrack;
//     }
  }
  
  const float mADC = SAMPAProcessing::getADCSaturation(totalADC);
  
  if(mADC > 0) {
    TClonesArray &clref = *output;
    new(clref[clref.GetEntriesFast()]) Digit(-1, -1, cru, mADC, row, pad, timeBin, commonMode);
  }
}

void DigitPad::processCommonMode(int cru, int timeBin, int row, int pad)
{  
  for(auto &aADCCounts : mADCCounts) {
    mTotalChargePad += aADCCounts.getADC();
  }
}
