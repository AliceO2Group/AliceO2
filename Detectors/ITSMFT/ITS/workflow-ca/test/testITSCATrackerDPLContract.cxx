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

BOOST_AUTO_TEST_CASE(PublicationCompatibilityIsClearedOnEveryWorkflowExit)
{
  o2::its::ca::PublicationAdapter publication;
  o2::itsmft::tracking::ITSSharedClusterCompatibility compatibility;
  publication.adoptITSSharedClusterCompatibility(&compatibility);
  o2::itsmft::tracking::TimeFrame frame;
  o2::itsmft::IterationParameters parameters;
  for (bool fail : {false, true}) {
    BOOST_REQUIRE(publication.completeAccepted({}, parameters, frame, true));
    BOOST_REQUIRE(compatibility.isSealed());
    try {
      auto cleanup = publication.cleanupOnExit();
      BOOST_CHECK(!compatibility.isSealed());
      BOOST_REQUIRE(publication.completeAccepted({}, parameters, frame, true));
      BOOST_CHECK(compatibility.isSealed());
      if (fail) {
        throw std::runtime_error{"publication failure"};
      }
    } catch (const std::runtime_error&) {
      BOOST_CHECK(fail);
    }
    BOOST_CHECK(!compatibility.isSealed());
    BOOST_CHECK(compatibility.entries().empty());
  }
}
