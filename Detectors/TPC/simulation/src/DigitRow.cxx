#include "TPCSimulation/DigitRow.h"
#include "TPCSimulation/DigitPad.h"
#include "FairLogger.h"
using namespace AliceO2::TPC;

DigitRow::DigitRow(Int_t rowID, Int_t npads):
mRowID(rowID),
mPads(npads)
{}

DigitRow::~DigitRow()
{
  for(auto iterPad = mPads.begin(); iterPad != mPads.end(); ++iterPad) {
    delete (*iterPad);
  }
}

void DigitRow::setDigit(Int_t pad, Int_t time, Float_t charge)
{
  DigitPad *result = mPads[pad];
  if(result != nullptr){
    mPads[pad]->setDigit(time, charge);
  }
  else{
    mPads[pad] = new DigitPad(pad);
    mPads[pad]->setDigit(time, charge);
  }
}

void DigitRow::reset()
{
  for(auto iterPad = mPads.begin(); iterPad != mPads.end(); ++iterPad) {
    if((*iterPad) == nullptr) continue;
    (*iterPad)->reset();
  }
}

void DigitRow::fillOutputContainer(TClonesArray *output, Int_t cruID, Int_t rowID)
{
  for(auto iterPad = mPads.begin(); iterPad != mPads.end(); ++iterPad) {
    if((*iterPad) == nullptr) continue;
    (*iterPad)->fillOutputContainer(output, cruID, rowID, (*iterPad)->getPad());
  }
}
