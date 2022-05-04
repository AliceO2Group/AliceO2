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

/// @file   HypertrackingWriterSpec.cxx

#include <vector>
#include "StrangenessTrackingWorkflow/HypertrackingWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "CommonDataFormat/TimeStamp.h"
#include "CommonDataFormat/RangeReference.h"
#include "ReconstructionDataFormats/V0.h"
#include "ReconstructionDataFormats/Track.h"
#include "StrangenessTracking/HyperTracker.h"

using namespace o2::framework;

namespace o2
{
namespace strangeness_tracking
{
using V0 = o2::dataformats::V0;

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

using namespace o2::header;

DataProcessorSpec getHypertrackingWriterSpec()
{
  auto loggerV = [](std::vector<V0> const& v) {
    LOG(info) << "Hypertracker writer pulled " << v.size() << " v0s";
  };

  auto loggerT = [](std::vector<o2::track::TrackParCov> const& v) {
    LOG(info) << "Hypertracker writer pulled " << v.size() << " tracks";
  };

  auto inpV0ID = InputSpec{"v0s", "HYP", "V0S", 0};
  auto inpTrackID = InputSpec{"hypertrack", "HYP", "HYPERTRACKS", 0};
  auto inpChi2ID = InputSpec{"v0itschi2", "HYP", "CHI2", 0};
  auto inpHe3Att = InputSpec{"he3updates", "HYP", "HE3UPDATES",0};
  auto inpRefID = InputSpec{"itsrefs", "HYP", "ITSREFS", 0};

  return MakeRootTreeWriterSpec("hypertracking-writer",
                                "o2_hypertrack.root",
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with Hypertracks"},
                                BranchDefinition<std::vector<V0>>{inpV0ID, "V0s", loggerV},
                                BranchDefinition<std::vector<o2::track::TrackParCov>>{inpTrackID, "Hypertracks", loggerT},
                                BranchDefinition<std::vector<float>>{inpChi2ID, "ITSV0Chi2"},
                                BranchDefinition<std::vector<o2::strangeness_tracking::He3Attachments>>{inpHe3Att, "He3Updates"},
                                BranchDefinition<std::vector<int>>{inpRefID, "ITSTrackRefs"})();
}

} // namespace strangeness_tracking
} // namespace o2