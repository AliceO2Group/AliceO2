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

/// \file   StrangenessWriterSpec.cxx
/// \brief

#include <vector>
#include "GlobalTrackingWorkflow/StrangenessTrackingWriterSpec.h"

#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "CommonDataFormat/TimeStamp.h"
#include "CommonDataFormat/RangeReference.h"
#include "ReconstructionDataFormats/StrangeTrack.h"
#include "StrangenessTracking/StrangenessTracker.h"
#include "SimulationDataFormat/MCCompLabel.h"

using namespace o2::framework;

namespace o2
{
namespace strangeness_tracking
{
using StrangeTrack = dataformats::StrangeTrack;

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using LabelsType = std::vector<o2::MCCompLabel>;
using namespace o2::header;

DataProcessorSpec getStrangenessTrackingWriterSpec(bool useMC)
{
  auto loggerV = [](std::vector<StrangeTrack> const& v) {
    LOG(info) << "StrangenessTracker writer pulled " << v.size() << " strange tracks";
  };

  auto inpStTrkID = InputSpec{"strangetracks", "STK", "STRTRACKS", 0};
  auto inpClusAtt = InputSpec{"clusupdates", "STK", "CLUSUPDATES", 0};
  auto inpMCLab = InputSpec{"stkmclabels", "STK", "STRK_MC", 0};

  return MakeRootTreeWriterSpec("strangenesstracking-writer",
                                "o2_strange_tracks.root",
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with Strange Tracks"},
                                BranchDefinition<std::vector<StrangeTrack>>{inpStTrkID, "StrangeTracks", loggerV},
                                BranchDefinition<std::vector<o2::strangeness_tracking::ClusAttachments>>{inpClusAtt, "ClusUpdates"},
                                BranchDefinition<LabelsType>{inpMCLab, "StrangeTrackMCLab", (useMC ? 1 : 0), ""} // one branch if mc labels enabled
                                )();
}

} // namespace strangeness_tracking
} // namespace o2