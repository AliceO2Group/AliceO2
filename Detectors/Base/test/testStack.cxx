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
#include "DetectorsBase/TrackTransportUtils.h"
#include "SimulationDataFormat/BaseHits.h"
#include "TFile.h"
#include "TMCProcess.h"
#include "TGeoManager.h"
#include "TGeoNavigator.h"
#include "TGeoCache.h"
#include "TGeoMaterial.h"
#include "TGeoMedium.h"
#include "TGeoMatrix.h"
#include "TGeoVolume.h"
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
  void collectHits(int, fair::mq::Parts&, int&, bool) override {}
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

// An invalid index must not be offset when sub-events are merged
BOOST_AUTO_TEST_CASE(Offsetting_keeps_an_invalid_index_invalid)
{
  const int nprimaries = 5, primaryOffset = 10, secondaryOffset = 100;

  BOOST_CHECK_EQUAL(o2::base::Detector::offsetTrackIndex(3, nprimaries, primaryOffset, secondaryOffset), 13);
  BOOST_CHECK_EQUAL(o2::base::Detector::offsetTrackIndex(7, nprimaries, primaryOffset, secondaryOffset), 107);
  BOOST_CHECK_EQUAL(o2::base::Detector::offsetTrackIndex(-1, nprimaries, primaryOffset, secondaryOffset), -1);
}

BOOST_AUTO_TEST_CASE(Track_transport_features_match_birth_v1)
{
  using namespace o2::data::detail;
  TParticle p(211, 0, 0, -1, -1, -1, 0., -2., 1., 2.5, 0., 0., 3., 7.e-9);
  auto f = makeTrackTransportFeatures(p, 2212, trackTransportMediumCode("TPC_DriftGas2"));
  BOOST_REQUIRE_EQUAL(f.size(), 34);
  BOOST_CHECK(validTrackTransportFeatures(f));
  BOOST_CHECK_EQUAL(f[0], 1.f);
  BOOST_CHECK_CLOSE(f[1], 0.13957039f, 0.01f);
  BOOST_CHECK_CLOSE(f[2], std::sqrt(5.f + f[1] * f[1]) - f[1], 1.e-4f);
  BOOST_CHECK_EQUAL(f[3], 2.f);
  BOOST_CHECK_CLOSE(f[4], std::asinh(0.5f), 1.e-4f);
  BOOST_CHECK_CLOSE(f[5], -std::acos(-1.f) / 2.f, 1.e-4f);
  BOOST_CHECK_CLOSE(f[6], std::acos(1.f / std::sqrt(5.f)), 1.e-4f);
  BOOST_CHECK_EQUAL(f[7], 0.f);
  BOOST_CHECK_EQUAL(f[8], 0.f);
  BOOST_CHECK_EQUAL(f[9], 3.f);
  BOOST_CHECK_EQUAL(f[10], 0.f);
  BOOST_CHECK_CLOSE(f[11], 7.f, 1.e-4f);
  BOOST_CHECK_EQUAL(f[12], 4.f);
  BOOST_CHECK(f[13] > 0.f);
  BOOST_CHECK_CLOSE(f[14], 2.f / 0.15f * 100.f, 1.e-4f);
  BOOST_CHECK_EQUAL(f[15], 23.f);
  BOOST_CHECK_EQUAL(f[16], 128.f);
  BOOST_CHECK_EQUAL(f[17], 2.f);
  BOOST_CHECK_EQUAL(f[18], 1.f);
  BOOST_CHECK_EQUAL(f[19], 211.f);
  BOOST_CHECK_EQUAL(f[20], 211.f);
  BOOST_CHECK_EQUAL(f[22], 0.f);
  BOOST_CHECK_EQUAL(f[23], -2.f);
  BOOST_CHECK_EQUAL(f[24], 1.f);
  BOOST_CHECK_EQUAL(f[32], 2212.f);
  BOOST_CHECK_EQUAL(f[33], 2212.f);
  const auto displaced = makeTrackTransportFeatures(p, 0., 0.f, 1., 2., 3.);
  BOOST_CHECK_EQUAL(displaced[27], -1.f);
  BOOST_CHECK_EQUAL(displaced[28], -2.f);
  BOOST_CHECK_EQUAL(displaced[29], 0.f);
  BOOST_CHECK_EQUAL(makeTrackTransportFeatures(p, 0, 0.f)[18], 0.f);
  BOOST_CHECK_EQUAL(trackTransportMediumCode("PIPE_VACUUM"), 1.f);
  BOOST_CHECK_EQUAL(trackTransportMediumCode("TPC_Air"), 3.f);
  BOOST_CHECK_EQUAL(trackTransportMediumCode("other"), 0.f);

  p.SetPdgCode(22);
  f = makeTrackTransportFeatures(p, 0, 0.f);
  BOOST_CHECK_EQUAL(f[0], 0.f);
  BOOST_CHECK_EQUAL(f[12], 0.f);
  BOOST_CHECK_EQUAL(f[14], 0.f);
  p.SetPdgCode(-211);
  f = makeTrackTransportFeatures(p, 0, 0.f);
  BOOST_CHECK_EQUAL(f[0], -1.f);
  BOOST_CHECK_EQUAL(f[12], 4.f);
  p.SetProductionVertex(300., 0., 3., 0.);
  BOOST_CHECK_EQUAL(trackTransportZAtRadius(p, 40.), 3.);
}

BOOST_AUTO_TEST_CASE(Track_transport_passes_unused_nonfinite_inputs_to_graph)
{
  using namespace o2::data::detail;
  TParticle p(211, 0, -1, -1, -1, -1, 1., 0., 0., 1.1, 0., 0., 0., 0.);
  auto f = makeTrackTransportFeatures(p, 0, 0.f);
  BOOST_REQUIRE(validTrackTransportFeatures(f));
  f[3] = 21.f;
  BOOST_CHECK(validTrackTransportFeatures(f));
  f[3] = std::numeric_limits<float>::quiet_NaN();
  BOOST_CHECK(validTrackTransportFeatures(f));
  f[3] = std::numeric_limits<float>::infinity();
  BOOST_CHECK(validTrackTransportFeatures(f));
  f.pop_back();
  BOOST_CHECK(!validTrackTransportFeatures(f));
  p.SetMomentum(0., 0., 0., 0.);
  f = makeTrackTransportFeatures(p, 0, 0.f);
  BOOST_CHECK(std::isnan(f[4]));
  BOOST_CHECK(std::isnan(f[6]));
  BOOST_CHECK(std::isnan(f[15]));
  BOOST_CHECK(validTrackTransportFeatures(f)); // graph selects/checks its own inputs
}

BOOST_AUTO_TEST_CASE(Track_transport_accepts_existing_squeezed_nn_output)
{
  using o2::data::detail::validTrackTransportOutput;
  BOOST_CHECK(validTrackTransportOutput({{-1}}, 0));
  BOOST_CHECK(validTrackTransportOutput({{1}}, 0));
  BOOST_CHECK(validTrackTransportOutput({{-1, 1}}, 0));
  BOOST_CHECK(validTrackTransportOutput({{-1, 2}}, 1));
  BOOST_CHECK(!validTrackTransportOutput({{-1}}, 1));
  BOOST_CHECK(!validTrackTransportOutput({{-1, 1}}, 1));
  BOOST_CHECK(!validTrackTransportOutput({{-1, -1}}, 0));
  BOOST_CHECK(!validTrackTransportOutput({{2}}, 0));
  BOOST_CHECK(!validTrackTransportOutput({{}}, 0));
  BOOST_CHECK(!validTrackTransportOutput({}, 0));
  BOOST_CHECK(!validTrackTransportOutput({{-1}, {-1}}, 0));
  BOOST_CHECK(!validTrackTransportOutput({{-1, 1, 1}}, 0));
  BOOST_CHECK(!validTrackTransportOutput({{-1}}, -1));
}

BOOST_AUTO_TEST_CASE(Track_transport_class_one_rejects_and_invalid_scores_keep)
{
  using o2::data::detail::transportFromOnnxScore;
  BOOST_CHECK(transportFromOnnxScore(0.1f, 0.5f, false));
  BOOST_CHECK(!transportFromOnnxScore(0.9f, 0.5f, false));
  BOOST_CHECK(transportFromOnnxScore(0.5f, std::nextafter(0.5, 1.), false));
  BOOST_CHECK(transportFromOnnxScore(1.f, std::nextafter(1., 2.), false));
  BOOST_CHECK(!transportFromOnnxScore(0.f, 0.5f, true));
  BOOST_CHECK(transportFromOnnxScore(-1000.f, 0.5f, true));
  BOOST_CHECK(!transportFromOnnxScore(1000.f, 0.5f, true));
  BOOST_CHECK(transportFromOnnxScore(0.9f, 0.5f, false, true));
  for (const bool invert : {false, true}) {
    BOOST_CHECK(transportFromOnnxScore(std::numeric_limits<float>::quiet_NaN(), 0.5f, false, invert));
    BOOST_CHECK(transportFromOnnxScore(std::numeric_limits<float>::infinity(), 0.5f, true, invert));
    BOOST_CHECK(transportFromOnnxScore(2.f, 0.5f, false, invert));
    BOOST_CHECK(transportFromOnnxScore(-0.1f, 0.5f, false, invert));
  }
  BOOST_CHECK_THROW(transportFromOnnxScore(0.5f, -1.f, false), std::runtime_error);
  BOOST_CHECK_THROW(transportFromOnnxScore(0.5f, std::numeric_limits<float>::quiet_NaN(), false), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(Track_transport_private_navigator_preserves_transport_state)
{
  TGeoManager geometry("pruning_test", "birth medium lookup");
  auto* material = new TGeoMaterial("material", 0., 0., 0.);
  auto* air = new TGeoMedium("TPC_Air", 1, material);
  auto* gas = new TGeoMedium("TPC_DriftGas2", 2, material);
  auto* world = geometry.MakeBox("world", air, 100., 100., 100.);
  auto* inner = geometry.MakeBox("inner", gas, 10., 10., 10.);
  world->AddNode(inner, 1);
  geometry.SetTopVolume(world);
  geometry.CloseGeometry();
  auto* transportNode = geometry.FindNode(50., 0., 0.);
  auto* transportNavigator = geometry.GetCurrentNavigator();
  TGeoNavigator lookup(&geometry);
  lookup.BuildCache();
  lookup.GetCache()->BuildInfoBranch();
  lookup.CdTop();
  auto* birthNode = lookup.FindNode(0., 0., 0.);
  BOOST_REQUIRE(birthNode);
  BOOST_CHECK_EQUAL(o2::data::detail::trackTransportMediumCode(birthNode->GetVolume()->GetMedium()->GetName()), 2.f);
  BOOST_CHECK(geometry.GetCurrentNavigator() == transportNavigator);
  BOOST_CHECK(geometry.GetCurrentNode() == transportNode);
  BOOST_CHECK_EQUAL(transportNavigator->GetCurrentPoint()[0], 50.);
}
