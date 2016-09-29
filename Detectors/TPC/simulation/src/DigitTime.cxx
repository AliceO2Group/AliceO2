#include "TPCSimulation/DigitTime.h"
#include "TPCSimulation/DigitADC.h"
#include "TPCSimulation/Digit.h"
#include "TPCSimulation/Digitizer.h"
#include "TClonesArray.h"
#include "FairLogger.h"
using namespace AliceO2::TPC;

DigitTime::DigitTime(Int_t timeBin) :
mTimeBin(timeBin)
{}

DigitTime::~DigitTime()
{
  for(auto iterADC = mADCCounts.begin(); iterADC != mADCCounts.end(); ++iterADC){
    delete (*iterADC);
  }
}

void DigitTime::setDigit(Float_t charge)
{
  digitAdc = new DigitADC(charge);
  mADCCounts.push_back(digitAdc);
}

void DigitTime::reset()
{
  mADCCounts.clear();                                    // delete all elements in the vector
  std::vector<DigitADC*>(mADCCounts).swap(mADCCounts);   // make sure the memory is deallocated
}

void DigitTime::fillOutputContainer(TClonesArray *output, Int_t cruID, Int_t rowID, Int_t padID, Int_t timeBin)
{
  Float_t mCharge = 0;
  for(auto iterADC = mADCCounts.begin(); iterADC != mADCCounts.end(); ++iterADC){
    if((*iterADC) == nullptr) continue;
    mCharge += (*iterADC)->getADC();
  }
  
  Digitizer d;
  const Int_t mADC = d.ADCvalue(mCharge);
  
  if(mADC > 0){
    Digit *digit = new Digit(cruID, mADC, rowID, padID, timeBin);
    TClonesArray &clref = *output;
    new(clref[clref.GetEntriesFast()]) Digit(*(digit));
  }
}
