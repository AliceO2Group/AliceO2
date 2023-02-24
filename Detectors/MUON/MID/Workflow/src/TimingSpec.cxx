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

/// \file   MID/Workflow/src/TimingSpec.cxx
/// \brief  Device to synchronize MID clock with collision BC
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   04 Avril 2022

#include "MIDWorkflow/TimingSpec.h"

#include <vector>
#include <gsl/gsl>
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "DataFormatsMID/ROFRecord.h"
#include "MIDRaw/ElectronicsDelay.h"
#include "MIDWorkflow/ColumnDataSpecsUtils.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{

class TimingDeviceDPL
{
 public:
  TimingDeviceDPL(int16_t localToBC, std::vector<of::OutputSpec> outputSpecs) : mLocalToBC(localToBC)
  {
    mOutputs = specs::buildOutputs(outputSpecs);
  }

  void init(o2::framework::InitContext& ic)
  {
  }

  void run(o2::framework::ProcessingContext& pc)
  {

    auto inRofs = specs::getRofs(pc, "mid_timing_in");
    std::array<std::vector<ROFRecord>, NEvTypes> outRofs;

    for (size_t idx = 0; idx < NEvTypes; ++idx) {
      if (idx == static_cast<size_t>(EventType::Calib)) {
        // Delays do not apply to triggered events
        outRofs[idx].insert(outRofs[idx].end(), inRofs[idx].begin(), inRofs[idx].end());
      } else {
        for (auto rof : inRofs[idx]) {
          applyElectronicsDelay(rof.interactionRecord.orbit, rof.interactionRecord.bc, mLocalToBC, mMaxBunches);
          outRofs[idx].emplace_back(rof);
        }
      }
      pc.outputs().snapshot(mOutputs[idx], outRofs[idx]);
    }
  }

 private:
  int16_t mLocalToBC = 0;
  uint16_t mMaxBunches = constants::lhc::LHCMaxBunches;
  std::vector<of::Output> mOutputs;
};

of::DataProcessorSpec getTimingSpec(int localToBC, std::string_view inRofDesc)
{
  auto inputSpecs = specs::buildInputSpecs("mid_timing_in", "", inRofDesc, "", false);
  auto outputSpecs = specs::buildOutputSpecs("mid_timing_out", "TDATAROF");

  return of::DataProcessorSpec{
    "MIDTiming",
    {inputSpecs},
    {outputSpecs},
    of::AlgorithmSpec{of::adaptFromTask<o2::mid::TimingDeviceDPL>(static_cast<int16_t>(localToBC), outputSpecs)}};
}
} // namespace mid
} // namespace o2