#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <TFile.h>
#include <TTree.h>
#include <TStopwatch.h>
#include <memory>
#include <iostream>
#include "DataFormatsFV0/BCData.h"
#include "DataFormatsFV0/ChannelData.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"

#endif

#include "FV0Simulation/MCLabel.h"
#include "FairLogger.h"

/// Example of accessing the FV0 digits

void readFV0Digits(std::string digiFName = "fv0digits.root")
{

  std::unique_ptr<TFile> digiFile(TFile::Open(digiFName.c_str()));
  if (!digiFile || digiFile->IsZombie()) {
    LOG(ERROR) << "Failed to open input digits file " << digiFName;
    return;
  }

  TTree* digiTree = (TTree*)digiFile->Get("o2sim");
  if (!digiTree) {
    LOG(ERROR) << "Failed to get digits tree";
    return;
  }

  std::vector<o2::fv0::BCData> fv0BCData, *fv0BCDataPtr = &fv0BCData;
  std::vector<o2::fv0::ChannelData> fv0ChData, *fv0ChDataPtr = &fv0ChData;
  o2::dataformats::MCTruthContainer<o2::fv0::MCLabel>* labelsPtr = nullptr;

  digiTree->SetBranchAddress("FV0DigitBC", &fv0BCDataPtr);
  digiTree->SetBranchAddress("FV0DigitCh", &fv0ChDataPtr);
  if (digiTree->GetBranch("FV0DigitLabels")) {
    digiTree->SetBranchAddress("FV0DigitLabels", &labelsPtr);
  }

  for (int ient = 0; ient < digiTree->GetEntries(); ient++) {
    digiTree->GetEntry(ient);
    int nbc = fv0BCData.size();
    LOG(INFO) << "Entry " << ient << " : " << nbc << " BCs stored";
    int itrig = 0;

    for (int ibc = 0; ibc < nbc; ibc++) {
      const auto& bcd = fv0BCData[ibc];
      bcd.print();
      int chEnt = bcd.ref.getFirstEntry();
      for (int ic = 0; ic < bcd.ref.getEntries(); ic++) {
        const auto& chd = fv0ChData[chEnt++];
        chd.print();
      }
      // alternative way - not implemented yet, as 'channels' are currently filled with fixed 0:
      /*
      auto channels = bcd.getBunchChannelData(fv0ChData);
      int nch = channels.size();
      for (int ich = 0; ich < nch; ich++) {
        channels[ich].print();
      }*/

      if (labelsPtr) {
        const auto lbl = labelsPtr->getLabels(ibc);
        for (int lb = 0; lb < lbl.size(); lb++) {
          printf("Ch%3d ", lbl[lb].getChannel());
          printf("Src%3d ", lbl[lb].getSourceID());
          lbl[lb].print();
        }
      }
    }
  }
}
