#include "DigitRow.h"
#include "DigitPad.h"
#include "Mapper.h"

#include "FairLogger.h"
using namespace AliceO2::TPC;

DigitRow::DigitRow(Int_t rowID, Int_t npads):
mRowID(rowID),
mNPads(npads),
mPads(npads)
{}

DigitRow::~DigitRow(){
  for (int ipad = 0; ipad < mNPads; ipad++) {
    delete mPads[ipad];
  }
}

void DigitRow::SetDigit(Int_t pad, Int_t time, Float_t charge){
  DigitPad *result = mPads[pad];
  if(result != nullptr){
    mPads[pad]->SetDigit(time, charge);
  }
  else{
    mPads[pad] = new DigitPad(pad,1000);
    mPads[pad]->SetDigit(time, charge);
  }
}

void DigitRow::Reset(){
  for(std::vector<DigitPad*>::iterator iterPad = mPads.begin(); iterPad != mPads.end(); iterPad++) {
    if((*iterPad) == nullptr) continue;
    (*iterPad)->Reset();
  }
}

void DigitRow::FillOutputContainer(TClonesArray *output, Int_t cruID, Int_t rowID){
  for(std::vector<DigitPad*>::iterator iterPad = mPads.begin(); iterPad != mPads.end(); iterPad++) {
    if((*iterPad) == nullptr) continue;
    (*iterPad)->FillOutputContainer(output, cruID, rowID, (*iterPad)->GetPad());
  }
}
