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

// Gate 3 workflow-onboarding Slice 2: focused tests for the DPL input/output
// contract of o2::its::ca::getCATrackerSpec() -- MC/non-MC variants, and the
// hard requirement that no vertex-related OutputSpec (VERTICES,
// VERTICESROF, VERTICESMCTR, VERTICESMCPUR, or any fake substitute) is ever
// declared by this opt-in tracker-only workflow.

#define BOOST_TEST_MODULE ITSMFT ITSCATrackerDPLContract
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <string>

#include "Framework/DataProcessorSpec.h"
#include "Framework/DataSpecUtils.h"
#include "ITSCAWorkflow/CATrackerSpec.h"

using namespace o2::framework;

namespace
{
bool hasInput(const std::vector<InputSpec>& specs, const std::string& binding)
{
  return std::any_of(specs.begin(), specs.end(), [&binding](const InputSpec& s) { return s.binding == binding; });
}

bool hasOutput(const std::vector<OutputSpec>& specs, const std::string& desc)
{
  return std::any_of(specs.begin(), specs.end(),
                     [&desc](const OutputSpec& s) { return DataSpecUtils::describe(s).find(desc) != std::string::npos; });
}
} // namespace

BOOST_AUTO_TEST_CASE(NonMCContractHasNoLabelsInputOrMCOutputs)
{
  const auto spec = o2::its::ca::getCATrackerSpec({.useMC = false});

  BOOST_CHECK(hasInput(spec.inputs, "compClusters"));
  BOOST_CHECK(hasInput(spec.inputs, "patterns"));
  BOOST_CHECK(hasInput(spec.inputs, "ROframes"));
  BOOST_CHECK(hasInput(spec.inputs, "itscldict"));
  BOOST_CHECK(hasInput(spec.inputs, "itsTGeo")); // useGeom=false: geometry CCDB requested explicitly
  BOOST_CHECK(!hasInput(spec.inputs, "labels"));

  BOOST_CHECK(hasOutput(spec.outputs, "TRACKS"));
  BOOST_CHECK(hasOutput(spec.outputs, "TRACKCLSID"));
  BOOST_CHECK(hasOutput(spec.outputs, "ITSTrackROF"));
  BOOST_CHECK(!hasOutput(spec.outputs, "TRACKSMCTR"));
}

BOOST_AUTO_TEST_CASE(MCContractAddsLabelsInputAndMCOutput)
{
  const auto spec = o2::its::ca::getCATrackerSpec({.useMC = true});

  BOOST_CHECK(hasInput(spec.inputs, "labels"));
  BOOST_CHECK(hasOutput(spec.outputs, "TRACKSMCTR"));
}

BOOST_AUTO_TEST_CASE(UseGeomOmitsExplicitGeometryInput)
{
  const auto spec = o2::its::ca::getCATrackerSpec({.useMC = false, .useFullGeometry = true});
  BOOST_CHECK(!hasInput(spec.inputs, "itsTGeo"));
}

BOOST_AUTO_TEST_CASE(NoVertexRelatedOutputsArePresentEver)
{
  for (const bool useMC : {false, true}) {
    const auto spec = o2::its::ca::getCATrackerSpec({.useMC = useMC});
    for (const auto& out : spec.outputs) {
      const auto desc = DataSpecUtils::describe(out);
      BOOST_CHECK_MESSAGE(desc.find("VERTICES") == std::string::npos,
                          "unexpected vertex-related output present: " << desc);
      BOOST_CHECK_MESSAGE(desc.find("VERTEX") == std::string::npos,
                          "unexpected vertex-related output present: " << desc);
    }
  }
}

BOOST_AUTO_TEST_CASE(DeviceNameIsStable)
{
  const auto spec = o2::its::ca::getCATrackerSpec({.useMC = false});
  BOOST_CHECK_EQUAL(spec.name, "its-ca-tracker");
}

BOOST_AUTO_TEST_CASE(PublicationFlagsAreClearedOnEveryWorkflowExit)
{
  o2::its::ca::PublicationAdapter publication;
  o2::itsmft::tracking::TimeFrame frame;
  frame.getGenericTracks().resize(1);
  o2::itsmft::IterationParameters parameters;
  parameters.AllowSharingFirstCluster = false;
  const std::array<uint32_t, 1> indices{0};
  for (bool fail : {false, true}) {
    BOOST_REQUIRE(publication.completeAccepted(indices, parameters, frame, true));
    BOOST_REQUIRE_EQUAL(publication.sharedClusterFlags().size(), 1u);
    try {
      auto cleanup = publication.cleanupOnExit();
      BOOST_CHECK(publication.sharedClusterFlags().empty());
      BOOST_REQUIRE(publication.completeAccepted(indices, parameters, frame, true));
      BOOST_REQUIRE_EQUAL(publication.sharedClusterFlags().size(), 1u);
      BOOST_CHECK_EQUAL(publication.sharedClusterFlags()[0], 0);
      if (fail) {
        throw std::runtime_error{"publication failure"};
      }
    } catch (const std::runtime_error&) {
      BOOST_CHECK(fail);
    }
    BOOST_CHECK(publication.sharedClusterFlags().empty());
  }
}

BOOST_AUTO_TEST_CASE(PublicationFlagsRequireFinalCompletionAndAcceptedIndices)
{
  o2::its::ca::PublicationAdapter publication;
  o2::itsmft::tracking::TimeFrame frame;
  frame.getGenericTracks().resize(3);
  o2::itsmft::IterationParameters parameters;
  parameters.AllowSharingFirstCluster = false;
  const std::array<uint32_t, 1> first{0}, last{2}, outOfRange{3};
  BOOST_REQUIRE(publication.completeAccepted(first, parameters, frame, false));
  BOOST_CHECK(publication.sharedClusterFlags().empty());
  BOOST_REQUIRE(publication.completeAccepted(last, parameters, frame, true));
  const auto flags = publication.sharedClusterFlags();
  BOOST_REQUIRE_EQUAL(flags.size(), 3u);
  BOOST_CHECK_EQUAL(flags[0], 0);
  BOOST_CHECK_EQUAL(flags[1], std::numeric_limits<uint8_t>::max());
  BOOST_CHECK_EQUAL(flags[2], 0);
  BOOST_CHECK(!publication.completeAccepted(last, parameters, frame, true));
  BOOST_CHECK(publication.sharedClusterFlags().empty());
  publication.reset();
  BOOST_CHECK(!publication.completeAccepted(outOfRange, parameters, frame, true));
  const std::array<uint32_t, 2> reversed{2, 0}, repeated{0, 0};
  BOOST_CHECK(!publication.completeAccepted(reversed, parameters, frame, true));
  BOOST_CHECK(!publication.completeAccepted(repeated, parameters, frame, true));
  BOOST_REQUIRE(publication.completeAccepted(first, parameters, frame, false));
  BOOST_REQUIRE(publication.completeAccepted({}, parameters, frame, true));
  BOOST_REQUIRE_EQUAL(publication.sharedClusterFlags().size(), 1u);
  BOOST_CHECK_EQUAL(publication.sharedClusterFlags()[0], 0);
}
