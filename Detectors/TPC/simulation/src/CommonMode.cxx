#include "TPCSimulation/CommonMode.h"
#include "TPCSimulation/Digit.h"
#include "TPCBase/CRU.h"
#include "TPCBase/Mapper.h"
#include "TPCSimulation/Digitizer.h"
#include "TPCSimulation/Constants.h"

using namespace AliceO2::TPC;

CommonMode::CommonMode() :
mCRU(),
mTimeBin(),
mCommonMode()
{}

CommonMode::CommonMode(Int_t cru, Int_t timeBin, Float_t commonMode) :
mCRU(cru),
mTimeBin(timeBin),
mCommonMode(commonMode)
{}

CommonMode::~CommonMode() {}


Float_t CommonMode::computeCommonMode(std::vector<CommonMode> & summedChargesContainer, std::vector<CommonMode> & commonModeContainer) { 
  
  const Mapper& mapper = Mapper::instance();
  
  for(auto &aSummedCharges : summedChargesContainer) {
    const int currentCRU = aSummedCharges.getCRU();
    const int timeBin = aSummedCharges.getTimeBin();
    float commonModeSignal = -(aSummedCharges.getCommonMode());
    
    CRU cru(currentCRU);
    int sector = int(cru.sector().getSector());
    const int gemStack = int(cru.gemStack());
    
    int cruLower =0;
    int cruUpper = 9;
    unsigned short nPads = 0;
    
    if(gemStack == 0) {cruLower = 0; cruUpper = 3; nPads = mapper.getPadsInIROC();}
    else if(gemStack ==1) {cruLower = 4; cruUpper = 5; nPads = mapper.getPadsInOROC1();}
    else if(gemStack ==2) {cruLower = 6; cruUpper = 7; nPads = mapper.getPadsInOROC2();}
    else {cruLower = 8; cruUpper = 9; nPads = mapper.getPadsInOROC3();}
    
    cruLower += sector*10.;
    cruUpper += sector*10.;
    
    for(auto &bSummedCharges : summedChargesContainer){
      // sum up charges in other CRUs of that GEM stack, but avoiding any double counting of signals from the very same CRU
      if(bSummedCharges.getCRU() >= cruLower && bSummedCharges.getCRU() <= cruUpper && bSummedCharges.getCRU() != currentCRU) {
        if(timeBin == bSummedCharges.getTimeBin()) {
          commonModeSignal -= bSummedCharges.getCommonMode();
        }
        if(bSummedCharges.getCRU() > cruUpper) break;
      }
    }
    
    commonModeSignal *= CPAD/(float(nPads)*CPAD);
    Digitizer d;
    const Float_t CommonModeADC = d.ADCvalue(commonModeSignal);
    CommonMode commonModeSummed(currentCRU, timeBin, CommonModeADC);
    commonModeContainer.emplace_back(commonModeSummed);
  }  
}