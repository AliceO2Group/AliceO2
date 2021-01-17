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
#include "Steer/MCKinematicsReader.h"
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

  o2::steer::MCKinematicsReader mcreader(nameprefix, o2::steer::MCKinematicsReader::Mode::kMCKine);

  for (int eventID = 0; eventID < mcbr->GetEntries(); ++eventID) {
    mcbr->GetEntry(eventID);
    refbr->GetEntry(eventID);
    LOG(DEBUG) << "-- Entry --" << eventID;
    LOG(DEBUG) << "Have " << mctracks->size() << " tracks";
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
      LOG(DEBUG) << " track " << ti << "\t" << t.getMotherTrackId() << " hits " << t.hasHits();
      ti++;
    }

    LOG(DEBUG) << "Have " << trackidsinTPC.size() << " tracks with hits in TPC";
    LOG(DEBUG) << "Have " << trackrefs->size() << " track refs";

    // check correct working of MCKinematicsReader
    bool havereferences = trackrefs->size();
    if (havereferences) {
      for (auto& trackID : trackidsinTPC) {
        auto trackrefs = mcreader.getTrackRefs(eventID, trackID);
        assert(trackrefs.size() > 0);
        LOG(DEBUG) << " Track " << trackID << " has " << trackrefs.size() << " TrackRefs";
        for (auto& ref : trackrefs) {
          assert(ref.getTrackID() == trackID);
        }
      }
    }
  }
  LOG(INFO) << "STACK TEST SUCCESSFULL\n";
  return 0;
}
