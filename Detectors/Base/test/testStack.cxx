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

#define BOOST_TEST_MODULE Test MCStack class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "DetectorsBase/Detector.h"
#include "DetectorsBase/Stack.h"
#include "SimulationDataFormat/BaseHits.h"
#include "TFile.h"
#include "TMCProcess.h"
#include "TRefArray.h"
#include <map>
#include <string>
#include <vector>

using namespace o2;

// unit tests on MC stack
BOOST_AUTO_TEST_CASE(Stack_test)
{
  o2::data::Stack st;
  int a;
  TMCProcess proc{kPPrimary};
  // add a 2 primary particles
  st.PushTrack(1, -1, 0, 0, 0., 0., 10., 5., 5., 5., 0.1, 0., 0., 0., proc, a, 1., 1);
  st.PushTrack(1, -1, 0, 0, 0., 0., 10., 5., 5., 5., 0.1, 0., 0., 0., proc, a, 1., 1);
  BOOST_CHECK(st.getPrimaries().size() == 2);

  {
    // serialize it
    TFile f("StackOut.root", "RECREATE");
    f.WriteObject(&st, "Stack");
    f.Close();
  }

  {
    o2::data::Stack* inst = nullptr;
    TFile f("StackOut.root", "OPEN");
    f.GetObject("Stack", inst);
    BOOST_CHECK(inst->getPrimaries().size() == 2);
  }
}

// convenience wrapper to push a track and return the assigned trackID
static int pushTrack(o2::data::Stack& st, int parentId, TMCProcess proc)
{
  int trackId;
  st.PushTrack(1, parentId, 0, 0., 0., 0., 10., 5., 5., 5., 0.1, 0., 0., 0., proc, trackId, 1., 1);
  return trackId;
}

// unit test for the radioactive-decay ancestry query
BOOST_AUTO_TEST_CASE(Stack_isFromRadDecay_test)
{
  o2::data::Stack st;

  // two primaries; note that primaries do not enter mParticles, only secondaries do
  const auto prim0 = pushTrack(st, -1, kPPrimary);
  const auto prim1 = pushTrack(st, -1, kPPrimary);

  // a radioactive decay product of the second primary, and its descendants.
  // this is deliberately the *first* secondary of the primary, so that it lands
  // in the first entry of the particle buffer
  const auto radDecay = pushTrack(st, prim1, kPRadDecay);
  const auto radChild = pushTrack(st, radDecay, kPHadronic);
  const auto radGrandChild = pushTrack(st, radChild, kPHadronic);

  // a plain secondary of the second primary: no radioactive decay anywhere in its history
  const auto ordinary = pushTrack(st, prim1, kPHadronic);

  // primaries can never come from a radioactive decay
  BOOST_CHECK(!st.isFromRadDecay(prim0));
  BOOST_CHECK(!st.isFromRadDecay(prim1));

  // a secondary whose ancestry ends in a primary must terminate the search with false
  BOOST_CHECK(!st.isFromRadDecay(ordinary));

  // directly and indirectly from a radioactive decay
  BOOST_CHECK(st.isFromRadDecay(radDecay));
  BOOST_CHECK(st.isFromRadDecay(radChild));
  BOOST_CHECK(st.isFromRadDecay(radGrandChild));

  // out-of-range track IDs are rejected rather than looked up
  BOOST_CHECK(!st.isFromRadDecay(-1));
  BOOST_CHECK(!st.isFromRadDecay(1000000000));
}

namespace
{
// A test detector to exercise hit creation and its interaction with the MCStack
class TestDetector : public o2::base::Detector
{
 public:
  // the name is turned into a DetID, so it has to be one of the real detectors
  TestDetector() : o2::base::Detector("ITS", true) {}

  void updateHitTrackIndices(std::map<int, int> const& indexmapping) override
  {
    for (auto& hit : mHits) {
      hit.SetTrackID(updatedTrackIndex(indexmapping, hit.GetTrackID()));
    }
  }

  std::vector<o2::BaseHit> mHits;

  // rest of the interface, unused here
  std::string getHitBranchNames(int) const override { return {}; }
  void attachHits(fair::mq::Channel&, fair::mq::Parts&) override {}
  void fillHitBranch(TTree&, fair::mq::Parts&, int&) override {}
  void collectHits(int, fair::mq::Parts&, int&) override {}
  void mergeHitEntriesAndFlush(int, TTree&, std::vector<int> const&, std::vector<int> const&,
                               std::vector<int> const&) override {}
  void mergeHitEntries(TTree&, TTree&, std::vector<int> const&, std::vector<int> const&,
                       std::vector<int> const&) override {}
  void InitializeO2Detector() override {}
  void initializeLate() override {}
  Bool_t ProcessHits(FairVolume* = nullptr) override { return kFALSE; }
  void Register() override {}
  void Reset() override {}
  void ConstructGeometry() override {}
};

// Transport one primary with n secondaries, so that the stack builds its mapping
void transportOnePrimary(o2::data::Stack& st, int nsecondaries)
{
  int ntr = 0;
  st.PushTrack(1, -1, 11, 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., kPPrimary, ntr, 1., 1);
  st.SetCurrentTrack(0);
  for (int i = 0; i < nsecondaries; ++i) {
    st.PushTrack(1, 0, 11, 0., 0., 0.1, 0.1, 0., 0., 0., 0., 0., 0., 0., kPHadronic, ntr, 1., 1);
  }
  st.FinishPrimary();
}
} // namespace

// A pruned track has no entry in the mapping
BOOST_AUTO_TEST_CASE(Unmapped_trackID_yields_invalid_index)
{
  const std::map<int, int> indexmapping{{0, 0}, {1, 1}};

  BOOST_CHECK_EQUAL(o2::base::Detector::updatedTrackIndex(indexmapping, 1), 1);
  BOOST_CHECK_EQUAL(o2::base::Detector::updatedTrackIndex(indexmapping, 99), -1);
}

// The mapping is per event and must not survive Reset()
BOOST_AUTO_TEST_CASE(Stack_does_not_reuse_index_map_of_previous_event)
{
  TestDetector det;
  TRefArray detlist;
  detlist.Add(&det);

  o2::data::Stack st;
  transportOnePrimary(st, 20); // event 1: trackIDs 0 to 20
  st.UpdateTrackIndex(&detlist);
  st.Reset();

  transportOnePrimary(st, 1); // event 2: trackIDs 0 and 1 only
  det.mHits.emplace_back(15); // only valid in event 1
  st.UpdateTrackIndex(&detlist);

  BOOST_CHECK_EQUAL(det.mHits[0].GetTrackID(), -1);
}
