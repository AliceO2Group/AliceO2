// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// Executable to check functioning of stack
// Analyses kinematics and track references of a kinematics file

#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCUtils.h"
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
#include <fairlogger/Logger.h>
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "ITSMFTSimulation/Hit.h"
#include <unordered_map>

int main(int argc, char** argv)
{
  const char* nameprefix = argv[1];

  fair::Logger::SetConsoleSeverity("DEBUG");
  TFile f(o2::base::NameConf::getMCKinematicsFileName(nameprefix).c_str());

  LOG(debug) << "Checking input file :" << f.GetPath();

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

  // when present we also read some hits for ITS to test consistency of trackID assignments
  TFile hitf(o2::base::DetectorNameConf::getHitsFileName(o2::detectors::DetID::ITS, nameprefix).c_str());
  auto hittr = (TTree*)hitf.Get("o2sim");
  auto hitbr = hittr ? hittr->GetBranch("ITSHit") : nullptr;
  std::vector<o2::itsmft::Hit>* hits = nullptr;
  if (hitbr) {
    hitbr->SetAddress(&hits);
  }

  for (int eventID = 0; eventID < mcbr->GetEntries(); ++eventID) {
    mcbr->GetEntry(eventID);
    refbr->GetEntry(eventID);
    LOG(debug) << "-- Entry --" << eventID;
    LOG(debug) << "Have " << mctracks->size() << " tracks";

    std::unordered_map<int, bool> trackidsinITS_fromhits;
    if (hitbr) {
      hitbr->GetEntry(eventID);
      LOG(debug) << "Have " << hits->size() << " hits";

      // check that trackIDs from the hits are within range
      int maxid = 0;
      for (auto& h : *hits) {
        maxid = std::max(maxid, h.GetTrackID());
        trackidsinITS_fromhits[h.GetTrackID()] = true;
        assert(maxid < mctracks->size());
      }
    }

    int ti = 0;

    // record tracks that left a hit in TPC
    // (we know that these tracks should then have a TrackRef)
    std::vector<int> trackidsinTPC;
    std::vector<int> trackidsinITS;

    int primaries = 0;
    int physicalprimaries = 0;
    int secondaries = 0;
    for (auto& t : *mctracks) {
      // perform checks on the mass
      if (t.GetMass() < 0) {
        LOG(info) << "Mass not found for PDG " << t.GetPdgCode();
      }

      if (t.isSecondary()) {
        // check that mother indices are monotonic
        // for primaries, this may be different (for instance with Pythia8)
        assert(ti > t.getMotherTrackId());
      }
      if (t.leftTrace(o2::detectors::DetID::TPC)) {
        trackidsinTPC.emplace_back(ti);
      }
      if (t.leftTrace(o2::detectors::DetID::ITS)) {
        trackidsinITS.emplace_back(ti);
      }
      bool physicalPrim = o2::mcutils::MCTrackNavigator::isPhysicalPrimary(t, *mctracks);
      LOG(debug) << " track " << ti << "\t" << t.getMotherTrackId() << " hits " << t.hasHits() << " isPhysicalPrimary " << physicalPrim;
      if (t.isPrimary()) {
        primaries++;
      } else {
        secondaries++;
      }
      if (physicalPrim) {
        physicalprimaries++;
      }
      ti++;
    }

    if (hitbr) {
      assert(trackidsinITS.size() == trackidsinITS_fromhits.size());
      for (auto id : trackidsinITS) {
        assert(trackidsinITS_fromhits[id] == true);
      }
    }

    LOG(debug) << "Have " << trackidsinTPC.size() << " tracks with hits in TPC";
    LOG(debug) << "Have " << trackrefs->size() << " track refs";
    LOG(info) << "Have " << primaries << " primaries and " << physicalprimaries << " physical primaries";

    // check correct working of MCKinematicsReader
    bool havereferences = trackrefs->size();
    if (havereferences) {
      for (auto& trackID : trackidsinTPC) {
        auto trackrefs = mcreader.getTrackRefs(eventID, trackID);
        assert(trackrefs.size() > 0);
        LOG(debug) << " Track " << trackID << " has " << trackrefs.size() << " TrackRefs";
        for (auto& ref : trackrefs) {
          assert(ref.getTrackID() == trackID);
        }
      }
    }
  }
  LOG(info) << "STACK TEST SUCCESSFULL\n";
  return 0;
}
