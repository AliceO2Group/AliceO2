/// \file DigitPad.cxx
/// \author Andi Mathis, andreas.mathis@ph.tum.de

#include "TPCSimulation/DigitPad.h"
#include "TPCSimulation/SAMPAProcessing.h"
#include "TPCSimulation/Digit.h"

using namespace AliceO2::TPC;

DigitPad::DigitPad(Int_t pad)
  : mPad(pad),
    mTotalChargePad(0.),
    mADCCounts()
{}

DigitPad::~DigitPad()
{
  mADCCounts.resize(0);
  mTotalChargePad = 0;
}

void DigitPad::fillOutputContainer(TClonesArray *output, Int_t cru, Int_t timeBin, Int_t row, Int_t pad)
{  
  Float_t totalADC = 0;
  for(auto &aADCCounts : mADCCounts) {
    totalADC += aADCCounts.getADC();
  }
  
  const Float_t mADC = SAMPAProcessing::getADCSaturation(totalADC);
  
  if(mADC > 0) {
    TClonesArray &clref = *output;
    new(clref[clref.GetEntriesFast()]) Digit(-1, -1, cru, mADC, row, pad, timeBin);
  }
}

void DigitPad::fillOutputContainer(TClonesArray *output, Int_t cru, Int_t timeBin, Int_t row, Int_t pad, Float_t commonMode)
{
  Float_t totalADC = 0;
  Int_t MCEventIDOld = -1;
  Int_t MCTrackIDOld = -1;
  for(auto &aADCCounts : mADCCounts) {
    totalADC += aADCCounts.getADC();
//     Int_t currentMCEvent = aADCCounts.getMCEventID();
//     Int_t currentMCTrack = aADCCounts.getMCTrackID();
//     if(MCEventIDOld != currentMCEvent) {
//       MCEventIDOld = currentMCEvent;
//     }
//     if(MCTrackIDOld != currentMCTrack) {
//       MCTrackIDOld = currentMCTrack;
//     }
  }
  
  const Float_t mADC = SAMPAProcessing::getADCSaturation(totalADC);
  
  if(mADC > 0) {
    TClonesArray &clref = *output;
    new(clref[clref.GetEntriesFast()]) Digit(-1, -1, cru, mADC, row, pad, timeBin, commonMode);
  }
}

void DigitPad::processCommonMode(Int_t cru, Int_t timeBin, Int_t row, Int_t pad)
{  
  for(auto &aADCCounts : mADCCounts) {
    mTotalChargePad += aADCCounts.getADC();
  }
}
