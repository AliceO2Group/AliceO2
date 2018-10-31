#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <TFile.h>
#include <TTree.h>
#include <TStopwatch.h>
#include <memory>
#include "FairLogger.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITSMFTBase/Digit.h"
#include "SimulationDataFormat/RunContext.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"

/// Example of accessing the digits of MCEvent digitized with continous readout

void readITSDigits(std::string path = "./",
                   std::string digiFName = "itsdigits.root",
                   std::string runContextFName = "collisioncontext.root",
                   std::string mctruthFName = "o2sim.root")
{
  if (path.back() != '/') {
    path += '/';
  }

  std::unique_ptr<TFile> digiFile(TFile::Open((path + digiFName).c_str()));
  if (!digiFile || digiFile->IsZombie()) {
    LOG(ERROR) << "Failed to open input digits file " << (path + digiFName) << FairLogger::endl;
    return;
  }
  std::unique_ptr<TFile> rcFile(TFile::Open((path + runContextFName).c_str()));
  if (!rcFile || rcFile->IsZombie()) {
    LOG(ERROR) << "Failed to open runContext file " << (path + runContextFName) << FairLogger::endl;
    return;
  }

  TTree* digiTree = (TTree*)digiFile->Get("o2sim");
  if (!digiTree) {
    LOG(ERROR) << "Failed to get digits tree" << FairLogger::endl;
    return;
  }

  std::vector<o2::ITSMFT::Digit>* dv = nullptr;
  digiTree->SetBranchAddress("ITSDigit", &dv);
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* labels = nullptr;
  digiTree->SetBranchAddress("ITSDigitMCTruth", &labels);

  // ROF record entries in the digit tree
  auto rofRecVec = reinterpret_cast<vector<o2::ITSMFT::ROFRecord>*>(digiFile->GetObjectChecked("ITSDigitROF", "vector<o2::ITSMFT::ROFRecord>"));
  if (!rofRecVec) {
    LOG(ERROR) << "Failed to get ITS digits ROF Records" << FairLogger::endl;
    return;
  }

  // MCEvID -> ROFrecord references
  auto mc2rofVec = reinterpret_cast<vector<o2::ITSMFT::MC2ROFRecord>*>(digiFile->GetObjectChecked("ITSDigitMC2ROF", "vector<o2::ITSMFT::MC2ROFRecord>"));
  if (!rofRecVec) {
    LOG(WARNING) << "Did not find MCEvent to ROF records references" << FairLogger::endl;
    return;
  }

  // MC collisions record
  auto runContext = reinterpret_cast<o2::steer::RunContext*>(rcFile->GetObjectChecked("RunContext", "o2::steer::RunContext"));
  if (!runContext) {
    LOG(WARNING) << "Did not find RunContext" << FairLogger::endl;
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
      LOG(WARNING) << "Event was not digitized" << FairLogger::endl;
      continue;
    }
    const auto& m2r = (*mc2rofVec)[iev];
    printf("Digitized to ROF %d - %d, entry %d in ROFRecords\n", m2r.minROF, m2r.maxROF, m2r.rofRecordID);
    int rofEntry = m2r.rofRecordID;
    for (auto rof = m2r.minROF; rof <= m2r.maxROF; rof++) {
      const auto& rofrec = (*rofRecVec)[rofEntry];
      rofrec.print();

      // read 1st and last digit of concerned rof
      digiTree->GetEntry(rofrec.getROFEntry().getEvent());
      int dgid = rofrec.getROFEntry().getIndex();
      const auto& digit0 = (*dv)[dgid];
      const auto& labs0 = labels->getLabels(dgid);
      printf("1st digit of this ROF (Entry: %6d) :", dgid);
      digit0.print(std::cout);
      printf(" MCinfo: ");
      labs0[0].print();

      dgid = rofrec.getROFEntry().getIndex() + rofrec.getNROFEntries() - 1;
      const auto& digit1 = (*dv)[dgid];
      const auto& labs1 = labels->getLabels(dgid);
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
