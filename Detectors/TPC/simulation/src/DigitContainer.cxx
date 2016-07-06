#include "DigitContainer.h"
#include "DigitCRU.h"
#include "Digit.h"
#include "Mapper.h"
#include "CRU.h"

#include "FairLogger.h"

#include "TClonesArray.h"

using namespace AliceO2::TPC;

DigitContainer::DigitContainer():
mCRU(CRU::MaxCRU)
{}

DigitContainer::~DigitContainer(){
  for(int icru = 0; icru < CRU::MaxCRU; ++icru){
    delete mCRU[icru];
  }
}

void DigitContainer::Reset(){
  for(std::vector<DigitCRU*>::iterator iterCRU = mCRU.begin(); iterCRU != mCRU.end(); iterCRU++) {
    if((*iterCRU) == nullptr) continue;
    (*iterCRU)->Reset();
  }
}

void DigitContainer::AddDigit(Int_t cru, Int_t row, Int_t pad, Int_t time, Float_t charge){
  DigitCRU *result = mCRU[cru];
  if(result != nullptr){
    mCRU[cru]->SetDigit(row, pad, time, charge);
  }
  else{
    const Mapper& mapper = Mapper::instance();
    mCRU[cru] = new DigitCRU(cru, mapper.getPadRegionInfo(CRU(cru).region()).getNumberOfPadRows());
    mCRU[cru]->SetDigit(row, pad, time, charge);
  }
}


void DigitContainer::FillOutputContainer(TClonesArray *output){
  for(std::vector<DigitCRU*>::iterator iterCRU = mCRU.begin(); iterCRU != mCRU.end(); iterCRU++) {
    if((*iterCRU) == nullptr) continue;
    (*iterCRU)->FillOutputContainer(output, (*iterCRU)->GetCRUID());
  }
}
