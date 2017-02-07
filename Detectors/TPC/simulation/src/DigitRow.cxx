/// \file DigitRow.cxx
/// \author Andi Mathis, andreas.mathis@ph.tum.de

#include "TPCSimulation/DigitRow.h"
#include "TPCSimulation/DigitPad.h"

using namespace AliceO2::TPC;

DigitRow::DigitRow(int row, int npads)
  : mTotalChargeRow(0.)
  , mRow(row)
  , mPads(npads)
{}

DigitRow::~DigitRow()
{
  for(auto &aPad : mPads) {
    delete aPad;
  }
}

void DigitRow::setDigit(int eventID, int trackID, int pad, float charge)
{
  DigitPad *result = mPads[pad];
  if(result != nullptr) {
    mPads[pad]->setDigit(eventID, trackID, charge);
  }
  else{
    mPads[pad] = new DigitPad(pad);
    mPads[pad]->setDigit(eventID, trackID, charge);
  }
}


void DigitRow::fillOutputContainer(TClonesArray *output, int cru, int timeBin, int row)
{
  for(auto &aPad : mPads) {
    if(aPad == nullptr) continue;
    aPad->fillOutputContainer(output, cru, timeBin, row, aPad->getPad());
  }
}

void DigitRow::fillOutputContainer(TClonesArray *output, int cru, int timeBin, int row, float commonMode)
{
  for(auto &aPad : mPads) {
    if(aPad == nullptr) continue;
    aPad->fillOutputContainer(output, cru, timeBin, row, aPad->getPad(), commonMode);
  }
}

void DigitRow::processCommonMode(int cru, int timeBin, int row)
{
  for(auto &aPad : mPads) {
    if(aPad == nullptr) continue;
    aPad->processCommonMode(cru, timeBin, row, aPad->getPad());
    mTotalChargeRow += aPad->getTotalChargePad();
  }
}
