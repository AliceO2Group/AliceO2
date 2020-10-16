#include <string>
#include <TFile.h>
#include <TTree.h>
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "ZDCBase/Constants.h"
#include "ZDCBase/ModuleConfig.h"
#include "ZDCSimulation/Digitizer.h"
#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/ChannelData.h"
#include "ZDCSimulation/Digits2Raw.h"
#include "ZDCSimulation/MCLabel.h"
#include "FairLogger.h"

using namespace o2::zdc;

//ClassImp(Digits2Raw);
void Digits2Raw::readDigits(const std::string& outDir, const std::string& fileDigitsName)
{
  if(!mModuleConfig){
    LOG(ERROR) << "Missing configuration object";
    return;
  }

  std::string outd = outDir;
  if (outd.back() != '/') {
    outd += '/';
  }

  std::unique_ptr<TFile> digiFile(TFile::Open(fileDigitsName.c_str()));
  if (!digiFile || digiFile->IsZombie()) {
    LOG(ERROR) << "Failed to open input digits file " << fileDigitsName;
    return;
  }

  TTree* digiTree = (TTree*)digiFile->Get("o2sim");
  if (!digiTree) {
    LOG(ERROR) << "Failed to get digits tree";
    return;
  }

  o2::dataformats::MCTruthContainer<o2::zdc::MCLabel>* labelsPtr = nullptr;

  if (digiTree->GetBranch("ZDCDigitBC")) {
    digiTree->SetBranchAddress("ZDCDigitBC", &mzdcBCDataPtr);
  } else {
    LOG(ERROR) << "Branch ZDCDigitBC is missing";
    return;
  }

  if (digiTree->GetBranch("ZDCDigitCh")) {
    digiTree->SetBranchAddress("ZDCDigitCh", &mzdcChDataPtr);
  } else {
    LOG(ERROR) << "Branch ZDCDigitCh is missing";
    return;
  }

  if (digiTree->GetBranch("ZDCDigitLabels")) {
    digiTree->SetBranchAddress("ZDCDigitLabels", &labelsPtr);
    LOG(INFO) << "Branch ZDCDigitLabels is connected";
  } else {
    LOG(INFO) << "Branch ZDCDigitLabels is missing";
  }

  for (int ient = 0; ient < digiTree->GetEntries(); ient++) {
    digiTree->GetEntry(ient);
    int nbc = mzdcBCData.size();
    LOG(INFO) << "Entry " << ient << " : " << nbc << " BCs stored";
    for (int ibc = 0; ibc < nbc; ibc++) {
    }
  }
  digiFile->Close();
}

void Digits2Raw::setTriggerMask(){
  mTriggerMask=0;
  mPrintTriggerMask="";
  for(Int_t im=0; im<NModules; im++){
    if(im>0)mPrintTriggerMask+=" ";
    mPrintTriggerMask+=to_string(im);
    mPrintTriggerMask+="[";
    for(UInt_t ic=0; ic<NChPerModule; ic++){
      if(mModuleConfig->modules[im].trigChannel[ic]){
	UInt_t tmask=0x1<<(im*NChPerModule+ic);
	trigger_mask=trigger_mask|tmask;
	mPrintTriggerMask+="T";
      }else{
	mPrintTriggerMask+=" ";
      }
    }
    mPrintTriggerMask+="]";
    UInt_t mytmask=trigger_mask>>(im*NChPerModule);
    printf("Trigger mask for module %d 0123 %s%s%s%s\n",im,
      mytmask&0x1?"T":"N",
      mytmask&0x2?"T":"N",
      mytmask&0x4?"T":"N",
      mytmask&0x8?"T":"N");
  }
  printf("trigger_mask=0x%08x %s\n",trigger_mask,mPrintTriggerMask.data());
}

void Digits2Raw::convertDigits(int ibc){
  // Orbit and bunch crossing identifiers
  const auto& bcd = mzdcBCData[ibc];
  UShort_t bc=bcd.ir.bc;
  UInt_t orbit=bcd.ir.orbit;
}
