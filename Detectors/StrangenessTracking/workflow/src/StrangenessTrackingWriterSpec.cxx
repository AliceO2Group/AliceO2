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
#include "StrangenessTrackingWorkflow/StrangenessTrackingWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "CommonDataFormat/TimeStamp.h"
#include "CommonDataFormat/RangeReference.h"
#include "ReconstructionDataFormats/V0.h"
#include "ReconstructionDataFormats/Track.h"
#include "StrangenessTracking/StrangenessTracker.h"

using namespace o2::framework;

namespace o2
{
namespace strangeness_tracking
{
using V0 = o2::dataformats::V0;

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

using namespace o2::header;

DataProcessorSpec getStrangenessTrackingWriterSpec()
{
  auto loggerV = [](std::vector<StrangeTrack> const& v) {
    LOG(info) << "StrangenessTracker writer pulled " << v.size() << " strange tracks";
  };

  auto inpStTrkID = InputSpec{"strangetracks", "STK", "STRTRACKS", 0};
  auto inpClusAtt = InputSpec{"clusupdates", "STK", "CLUSUPDATES", 0};

  return MakeRootTreeWriterSpec("strangenesstracking-writer",
                                "o2_strange_tracks.root",
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with Strange Tracks"},
                                BranchDefinition<std::vector<StrangeTrack>>{inpStTrkID, "StrangeTracks", loggerV},
                                BranchDefinition<std::vector<o2::strangeness_tracking::ClusAttachments>>{inpClusAtt, "ClusUpdates"})();
}

} // namespace strangeness_tracking
} // namespace o2