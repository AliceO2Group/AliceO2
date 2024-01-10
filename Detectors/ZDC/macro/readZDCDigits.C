#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <TFile.h>
#include <TTree.h>
#include <TStopwatch.h>
#include <memory>
#include <iostream>
#include "DataFormatsZDC/ChannelData.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"

#endif

#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/MCLabel.h"
#include "Framework/Logger.h"

/// Example of accessing the ZDC digits

void readZDCDigits(std::string digiFName = "zdcdigits.root")
{

  std::unique_ptr<TFile> digiFile(TFile::Open(digiFName.c_str()));
  if (!digiFile || digiFile->IsZombie()) {
    LOG(error) << "Failed to open input digits file " << digiFName;
    return;
  }

  TTree* digiTree = (TTree*)digiFile->Get("o2sim");
  if (!digiTree) {
    LOG(error) << "Failed to get digits tree";
    return;
  }

  std::vector<o2::zdc::BCData> zdcBCData, *zdcBCDataPtr = &zdcBCData;
  std::vector<o2::zdc::ChannelData> zdcChData, *zdcChDataPtr = &zdcChData;
  o2::dataformats::MCTruthContainer<o2::zdc::MCLabel>* labelsPtr = nullptr;

  digiTree->SetBranchAddress("ZDCDigitBC", &zdcBCDataPtr);
  digiTree->SetBranchAddress("ZDCDigitCh", &zdcChDataPtr);
  if (digiTree->GetBranch("ZDCDigitLabels")) {
    digiTree->SetBranchAddress("ZDCDigitLabels", &labelsPtr);
  }

  for (int ient = 0; ient < digiTree->GetEntries(); ient++) {
    digiTree->GetEntry(ient);
    int nbc = zdcBCData.size();
    LOG(info) << "Entry " << ient << " : " << nbc << " BCs stored";
    int itrig = 0;

    for (int ibc = 0; ibc < nbc; ibc++) {
      const auto& bcd = zdcBCData[ibc];
      if (bcd.triggers) {
        LOG(info) << "Triggered BC " << itrig++;
      } else {
        LOG(info) << "Non-Triggered BC ";
      }
      bcd.print();
      //
      auto channels = bcd.getBunchChannelData(zdcChData);
      int nch = channels.size();
      for (int ich = 0; ich < nch; ich++) {
        channels[ich].print();
      }
      /* // alternative way:
      int chEnt = bcd.ref.getFirstEntry();
      for (int ic = 0; ic < bcd.ref.getEntries(); ic++) {
        const auto& chd = zdcChData[chEnt++];
        chd.print();
      }
      */
      if (labelsPtr) {
        const auto lbl = labelsPtr->getLabels(ibc);
        for (int lb = 0; lb < lbl.size(); lb++) {
          printf("Ch%d ", lbl[lb].getChannel());
          lbl[lb].print();
        }
      }
    }
  }
}
