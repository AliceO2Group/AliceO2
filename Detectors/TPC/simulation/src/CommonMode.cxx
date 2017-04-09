/// \file CommonMode.cxx
/// \brief Implementation of the Common Mode computation
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCSimulation/CommonMode.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/CRU.h"
#include "TPCSimulation/Constants.h"
#include "TPCSimulation/SAMPAProcessing.h"

using namespace o2::TPC;

CommonMode::CommonMode()
  : mCRU()
  , mTimeBin()
  , mCommonMode()
{}

CommonMode::CommonMode(int cru, int timeBin, float commonMode)
  : mCRU(cru)
  , mTimeBin(timeBin)
  , mCommonMode(commonMode)
{}

CommonMode::~CommonMode() 
= default;


float CommonMode::computeCommonMode(std::vector<CommonMode> & summedChargesContainer, std::vector<CommonMode> & commonModeContainer)
{
  const Mapper& mapper = Mapper::instance();

  /// Loop over the full container and check whether CRUs on the same ROC have a signal in the same time bin
  for(auto &aSummedCharges : summedChargesContainer) {
    const int currentCRU = aSummedCharges.getCRU();
    const int timeBin = aSummedCharges.getTimeBin();
    float commonModeSignal = aSummedCharges.getCommonMode();

    CRU cru(currentCRU);
    const int sector = int(cru.sector().getSector());
    const int gemStack = int(cru.gemStack());

    int cruLower = 0;
    int cruUpper = 9;
    unsigned short nPads = 0;

    /// Identify the number of the adjacent CRUs and the total number of pads for a given CRU
    if(gemStack == 0) {
      cruLower = 0;
      cruUpper = 3;
      nPads = mapper.getPadsInIROC();
    }
    else if(gemStack ==1) {
      cruLower = 4;
      cruUpper = 5;
      nPads = mapper.getPadsInOROC1();
    }
    else if(gemStack ==2) {
      cruLower = 6;
      cruUpper = 7;
      nPads = mapper.getPadsInOROC2();
    }
    else {
      cruLower = 8;
      cruUpper = 9;
      nPads = mapper.getPadsInOROC3();
    }

    /// scale the identifiers of the adjacent CRU (in Sector 0) to the actual sector
    cruLower += sector*10.;
    cruUpper += sector*10.;

    for(auto &bSummedCharges : summedChargesContainer){
      /// sum up charges in other CRUs of that GEM ROC, but avoiding any double counting of signals from the very same CRU
      if(bSummedCharges.getCRU() >= cruLower && bSummedCharges.getCRU() <= cruUpper && bSummedCharges.getCRU() != currentCRU) {
        if(timeBin == bSummedCharges.getTimeBin()) {
          commonModeSignal += bSummedCharges.getCommonMode();
        }
      }
      if(bSummedCharges.getCRU() > cruUpper) break;
    }

    commonModeSignal /= static_cast<float>(nPads); /// the formula below can be simplified when there is no external capacitance on the ROC
    //commonModeSignal *= CPAD/(float(nPads)*CPAD);
    CommonMode commonModeSummed(currentCRU, timeBin, commonModeSignal);
    commonModeContainer.emplace_back(commonModeSummed);
  }
}
