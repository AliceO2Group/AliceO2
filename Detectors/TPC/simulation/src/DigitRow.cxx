/// \file DigitRow.cxx
/// \author Andi Mathis, andreas.mathis@ph.tum.de

#include "TPCSimulation/DigitRow.h"
#include "TPCSimulation/DigitPad.h"

using namespace AliceO2::TPC;

DigitRow::DigitRow(int row, int npads)
  : mRow(row)
  , mPads(npads)
{}

DigitRow::~DigitRow()
{
  mRow = 0;
  mPads.resize(0);
}

void DigitRow::setDigit(int eventID, int trackID, int pad, float charge)
{
  DigitPad *result =  mPads[pad].get();
  if(result != nullptr) {
    mPads[pad]->setDigit(eventID, trackID, charge);
  }
  else{
    mPads[pad] = std::unique_ptr<DigitPad> (new DigitPad(pad));
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