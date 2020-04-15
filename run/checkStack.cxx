// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// Executable to check functioning of stack
// Analyses kinematics and track references of a kinematics file

#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/Stack.h"
#include "SimulationDataFormat/TrackReference.h"
#include "TFile.h"
#include "TTree.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include "FairLogger.h"
#include "DetectorsCommonDataFormats/NameConf.h"

int main(int argc, char** argv)
{
  const char* nameprefix = argv[1];

  FairLogger::GetLogger()->SetLogScreenLevel("DEBUG");
  TFile f(o2::base::NameConf::getMCKinematicsFileName(nameprefix).c_str());

  LOG(DEBUG) << "Checking input file :" << f.GetPath();

  std::vector<o2::MCTrack>* mctracks = nullptr;
  auto tr = (TTree*)f.Get("o2sim");
  assert(tr);

  auto mcbr = tr->GetBranch("MCTrack");
  assert(mcbr);
  mcbr->SetAddress(&mctracks);

  std::vector<o2::TrackReference>* trackrefs = nullptr;
  auto refbr = tr->GetBranch("TrackRefs");
  assert(refbr);
  refbr->SetAddress(&trackrefs);

  o2::dataformats::MCTruthContainer<o2::TrackReference>* indexedtrackrefs = nullptr;
  auto irefbr = tr->GetBranch("IndexedTrackRefs");
  irefbr->SetAddress(&indexedtrackrefs);

  for (int i = 0; i < mcbr->GetEntries(); ++i) {
    mcbr->GetEntry(i);
    refbr->GetEntry(i);
    irefbr->GetEntry(i);
    LOG(DEBUG) << "-- Entry --" << i << "\n";
    LOG(DEBUG) << "Have " << mctracks->size() << " tracks \n";
    int ti = 0;

    // record tracks that left a hit in TPC
    // (we know that these tracks should then have a TrackRef)
    std::vector<int> trackidsinTPC;

    for (auto& t : *mctracks) {
      // check that mother indices are reasonable
      assert(ti > t.getMotherTrackId());
      if (t.leftTrace(o2::detectors::DetID::TPC)) {
        trackidsinTPC.emplace_back(ti);
      }
      LOG(DEBUG) << " track " << ti << "\t" << t.getMotherTrackId() << " hits " << t.hasHits() << "\n";
      ti++;
    }

    LOG(DEBUG) << "Have " << trackidsinTPC.size() << " tracks with hits in TPC\n";
    LOG(DEBUG) << "Have " << trackrefs->size() << " track refs \n";
    LOG(DEBUG) << "Have " << indexedtrackrefs->getIndexedSize() << " mapped tracks\n";
    LOG(DEBUG) << "Have " << indexedtrackrefs->getNElements() << " track refs\n";

    bool havereferences = trackrefs->size();
    if (havereferences) {
      for (auto& i : trackidsinTPC) {
        auto labels = indexedtrackrefs->getLabels(i);
        assert(labels.size() > 0);
        LOG(DEBUG) << " Track " << i << " has " << labels.size() << " TrackRefs "
                   << "\n";
        for (int j = 0; j < labels.size(); ++j) {
          LOG(DEBUG) << labels[j];
        }
      }
    }
  }
  LOG(INFO) << "STACK TEST SUCCESSFULL\n";
  return 0;
}
