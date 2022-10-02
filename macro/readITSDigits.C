#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <TFile.h>
#include <TTree.h>
#include <TStopwatch.h>
#include <memory>
#include <iostream>
#include <fairlogger/Logger.h>
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/Digit.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "SimulationDataFormat/MCCompLabel.h"

/// Example of accessing the digits of MCEvent digitized with continous readout

void readITSDigits(std::string path = "./",
                   std::string digiFName = "itsdigits.root",
                   std::string runContextFName = "collisioncontext.root")
{
  if (path.back() != '/') {
    path += '/';
  }

  std::unique_ptr<TFile> digiFile(TFile::Open((path + digiFName).c_str()));
  if (!digiFile || digiFile->IsZombie()) {
    LOG(error) << "Failed to open input digits file " << (path + digiFName);
    return;
  }
  std::unique_ptr<TFile> rcFile(TFile::Open((path + runContextFName).c_str()));
  if (!rcFile || rcFile->IsZombie()) {
    LOG(error) << "Failed to open runContext file " << (path + runContextFName);
    return;
  }

  TTree* digiTree = (TTree*)digiFile->Get("o2sim");
  if (!digiTree) {
    LOG(error) << "Failed to get digits tree";
    return;
  }

  std::vector<o2::itsmft::Digit>* dv = nullptr;
  digiTree->SetBranchAddress("ITSDigit", &dv);
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* labels = nullptr;
  o2::dataformats::IOMCTruthContainerView* labelROOTbuffer = nullptr;
  o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel> constlabels;

  // for backward compatibility we check what is stored in the file
  auto labelClass = digiTree->GetBranch("ITSDigitMCTruth")->GetClassName();
  bool oldlabelformat = false;
  if (TString(labelClass).Contains("IOMCTruth")) {
    // new format
    digiTree->SetBranchAddress("ITSDigitMCTruth", &labelROOTbuffer);
  } else {
    // old format
    digiTree->SetBranchAddress("ITSDigitMCTruth", &labels);
    oldlabelformat = true;
  }

  // ROF record entries in the digit tree
  std::vector<o2::itsmft::ROFRecord>* rofRecVec = nullptr;
  digiTree->SetBranchAddress("ITSDigitROF", &rofRecVec);

  // MCEvID -> ROFrecord references
  std::vector<o2::itsmft::MC2ROFRecord>* mc2rofVec = nullptr;
  digiTree->SetBranchAddress("ITSDigitMC2ROF", &mc2rofVec);

  digiTree->GetEntry(0);
  if (!oldlabelformat) {
    labelROOTbuffer->copyandflatten(constlabels);
  }

  // MC collisions record
  auto runContext = reinterpret_cast<o2::steer::DigitizationContext*>(rcFile->GetObjectChecked("DigitizationContext", "o2::steer::DigitizationContext"));
  if (!runContext) {
    LOG(warning) << "Did not find DigitizationContext";
    return;
  }

  auto intRecordsVec = runContext->getEventRecords(); // interaction record
  auto evPartsVec = runContext->getEventParts();      // event parts
  int nEvents = runContext->getNCollisions();

  for (int iev = 0; iev < nEvents; iev++) {
    const auto& collision = intRecordsVec[iev];
    const auto& evParts = evPartsVec[iev];

    int nmixed = evParts.size();
    printf("MCEvent %d made of %d MC parts: ", iev, (int)evParts.size());
    for (auto evp : evParts) {
      printf(" [src%d entry %d]", evp.sourceID, evp.entryID);
    }
    printf("\n");

    if (int(mc2rofVec->size()) <= iev || (*mc2rofVec)[iev].eventRecordID < 0) {
      LOG(warning) << "Event was not digitized";
      continue;
    }
    const auto& m2r = (*mc2rofVec)[iev];
    printf("Digitized to ROF %d - %d, entry %d in ROFRecords\n", m2r.minROF, m2r.maxROF, m2r.rofRecordID);
    int rofEntry = m2r.rofRecordID;
    for (auto rof = m2r.minROF; rof <= m2r.maxROF; rof++) {
      const auto& rofrec = (*rofRecVec)[rofEntry];
      rofrec.print();

      int dgid = rofrec.getFirstEntry();
      const auto& digit0 = (*dv)[dgid];
      const auto& labs0 = oldlabelformat ? labels->getLabels(dgid) : constlabels.getLabels(dgid);
      printf("1st digit of this ROF (Entry: %6d) :", dgid);
      digit0.print(std::cout);
      printf(" MCinfo: ");
      labs0[0].print();

      dgid = rofrec.getFirstEntry() + rofrec.getNEntries() - 1;
      const auto& digit1 = (*dv)[dgid];
      const auto& labs1 = oldlabelformat ? labels->getLabels(dgid) : constlabels.getLabels(dgid);
      printf("1st digit of this ROF (Entry: %6d) :", dgid);
      digit1.print(std::cout);
      printf(" MCinfo: ");
      labs1[0].print();
      //
      rofEntry++;
    }

    printf("\n");
  }
}

#endif
