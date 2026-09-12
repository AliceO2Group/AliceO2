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

#define BOOST_TEST_MODULE MFTCATrackerDPLContract
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <string>

#include "Framework/DataProcessorSpec.h"
#include "Framework/DataSpecUtils.h"
#include "MFTWorkflow/CATrackerSpec.h"

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

BOOST_AUTO_TEST_CASE(NonMCContractKeepsTheExistingMFTProducts)
{
  const auto spec = o2::mft::getCATrackerSpec({.useMC = false});
  BOOST_CHECK(hasInput(spec.inputs, "compClusters"));
  BOOST_CHECK(hasInput(spec.inputs, "patterns"));
  BOOST_CHECK(hasInput(spec.inputs, "ROframes"));
  BOOST_CHECK(hasInput(spec.inputs, "cldict"));
  BOOST_CHECK(hasInput(spec.inputs, "mftTGeo"));
  BOOST_CHECK(!hasInput(spec.inputs, "labels"));
  BOOST_CHECK(!hasInput(spec.inputs, "IRFramesITS"));

  BOOST_CHECK(hasOutput(spec.outputs, "TRACKS"));
  BOOST_CHECK(hasOutput(spec.outputs, "TRACKCLSID"));
  BOOST_CHECK(hasOutput(spec.outputs, "MFTTrackROF"));
  BOOST_CHECK(hasOutput(spec.outputs, "TRACKSEEDPAT"));
  BOOST_CHECK(!hasOutput(spec.outputs, "TRACKSMCTR"));
}

BOOST_AUTO_TEST_CASE(MCAndIRFrameContractRemainOptional)
{
  const auto spec = o2::mft::getCATrackerSpec({.useMC = true, .irFrames = o2::mft::ca::IRFrameSource::Upstream});
  BOOST_CHECK(hasInput(spec.inputs, "labels"));
  BOOST_CHECK(hasInput(spec.inputs, "IRFramesITS"));
  BOOST_CHECK(hasOutput(spec.outputs, "TRACKSMCTR"));
}

BOOST_AUTO_TEST_CASE(DeviceNameIsStableForWriterAssessmentAndAlignmentConsumers)
{
  const auto spec = o2::mft::getCATrackerSpec({.useMC = false, .geometry = o2::mft::ca::GeometrySource::Full});
  BOOST_CHECK_EQUAL(spec.name, "mft-ca-tracker");
  BOOST_CHECK(!hasInput(spec.inputs, "mftTGeo"));
}
