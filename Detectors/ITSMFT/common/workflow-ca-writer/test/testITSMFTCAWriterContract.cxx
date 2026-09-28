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

// Pin the shared ITS and MFT common-CA writer specifications.

#define BOOST_TEST_MODULE ITSMFT ITSMFTCAWriterContract
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <string>

#include "Framework/DataProcessorSpec.h"
#include "Framework/DataSpecUtils.h"
#include "ITSMFTCAWriter/ITSCATrackWriterSpec.h"
#include "ITSMFTCAWriter/MFTCATrackWriterSpec.h"

using namespace o2::framework;

namespace
{
bool hasInput(const std::vector<InputSpec>& specs, const std::string& binding)
{
  return std::any_of(specs.begin(), specs.end(), [&binding](const InputSpec& s) { return s.binding == binding; });
}

bool sameShape(const DataProcessorSpec& a, const DataProcessorSpec& b)
{
  if (a.name != b.name || a.inputs.size() != b.inputs.size() || a.outputs.size() != b.outputs.size()) {
    return false;
  }
  for (size_t i = 0; i < a.inputs.size(); ++i) {
    if (a.inputs[i].binding != b.inputs[i].binding || DataSpecUtils::describe(a.inputs[i]) != DataSpecUtils::describe(b.inputs[i])) {
      return false;
    }
  }
  return true;
}
} // namespace

BOOST_AUTO_TEST_CASE(ITSWriterSpecContract)
{
  const auto spec = o2::its::ca::getTrackWriterSpec(false);
  BOOST_CHECK_EQUAL(spec.name, "its-ca-track-writer");
  BOOST_CHECK(hasInput(spec.inputs, "tracks"));
  BOOST_CHECK(hasInput(spec.inputs, "trackClIdx"));
  BOOST_CHECK(hasInput(spec.inputs, "ROframes"));
  BOOST_CHECK(!hasInput(spec.inputs, "labels"));
}

BOOST_AUTO_TEST_CASE(ITSWriterSpecMCContractAddsLabels)
{
  const auto spec = o2::its::ca::getTrackWriterSpec(true);
  BOOST_CHECK(hasInput(spec.inputs, "labels"));
}

BOOST_AUTO_TEST_CASE(ITSWriterSpecIsDeterministicAcrossCallers)
{
  const auto first = o2::its::ca::getTrackWriterSpec(true);
  const auto second = o2::its::ca::getTrackWriterSpec(true);
  BOOST_CHECK(sameShape(first, second));
}

BOOST_AUTO_TEST_CASE(MFTWriterSpecContract)
{
  const auto spec = o2::mft::getTrackWriterSpec(false);
  BOOST_CHECK_EQUAL(spec.name, "mft-track-writer");
  BOOST_CHECK(hasInput(spec.inputs, "tracks"));
  BOOST_CHECK(hasInput(spec.inputs, "trackClIdx"));
  BOOST_CHECK(!hasInput(spec.inputs, "trackSeedPat"));
  BOOST_CHECK(hasInput(spec.inputs, "ROframes"));
  BOOST_CHECK(!hasInput(spec.inputs, "labels"));
}

BOOST_AUTO_TEST_CASE(MFTWriterSpecMCContractAddsLabels)
{
  const auto spec = o2::mft::getTrackWriterSpec(true);
  BOOST_CHECK(hasInput(spec.inputs, "labels"));
}

BOOST_AUTO_TEST_CASE(MFTWriterSpecDefaultUseCAIsFalseMatchingLegacyCaller)
{
  // RecoWorkflow.cxx (legacy o2-mft-reco-workflow) calls
  // getTrackWriterSpec(useMC) with useCA left at its default -- must stay
  // false so the legacy writer's own contract is unchanged.
  const auto legacyShape = o2::mft::getTrackWriterSpec(false);
  const auto explicitFalse = o2::mft::getTrackWriterSpec(false, false);
  BOOST_CHECK(sameShape(legacyShape, explicitFalse));
}

BOOST_AUTO_TEST_CASE(MFTWriterSpecIsDeterministicAcrossCallersUseCATrue)
{
  const auto first = o2::mft::getTrackWriterSpec(true, true);
  const auto second = o2::mft::getTrackWriterSpec(true, true);
  BOOST_CHECK(sameShape(first, second));
  BOOST_CHECK(hasInput(first.inputs, "trackSeedPat"));
}
