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

#include <vector>
#include "GlobalTrackingWorkflow/StrangenessTrackingWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "CommonDataFormat/TimeStamp.h"
#include "ReconstructionDataFormats/StrangeTrack.h"
#include "ReconstructionDataFormats/StrangenessTrackingStats.h"
#include "DataFormatsITS/TrackITS.h"
#include "StrangenessTracking/StrangenessTracker.h"
#include "SimulationDataFormat/MCCompLabel.h"

using namespace o2::framework;
namespace o2::strangeness_tracking
{
template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using LabelsType = std::vector<o2::MCCompLabel>;
using TracksType = std::vector<o2::dataformats::StrangeTrack>;
using TimesType = std::vector<o2::dataformats::TimeStampWithError<float, float>>;

DataProcessorSpec getStrangenessTrackingWriterSpec(bool useMC, bool enableExtensions, bool enableShortTracks)
{
  return MakeRootTreeWriterSpec(
    "strangenesstracking-writer", "o2_strange_tracks.root",
    MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with Strange Tracks"},
    BranchDefinition<TracksType>{InputSpec{"strangetracks", "GLO", "STRANGETRACKS", 0}, "StrangeTracks"},
    BranchDefinition<std::vector<ClusAttachments>>{InputSpec{"clusupdates", "GLO", "CLUSUPDATES", 0}, "ClusUpdates"},
    BranchDefinition<LabelsType>{InputSpec{"stkmclabels", "GLO", "STRANGETRACKS_MC", 0}, "StrangeTrackMCLab", useMC ? 1 : 0, ""},
    BranchDefinition<TracksType>{InputSpec{"extendedstracks", "GLO", "EXTSTRTRACKS", 0}, "ExtendedStrangeTracks", enableExtensions ? 1 : 0, ""},
    BranchDefinition<TimesType>{InputSpec{"extendedstracktimes", "GLO", "EXTSTRTIMES", 0}, "ExtendedStrangeTrackTimes", enableExtensions ? 1 : 0, ""},
    BranchDefinition<LabelsType>{InputSpec{"extendedstrack_mc", "GLO", "EXTSTRTRACKS_MC", 0}, "ExtendedStrangeTrackMCLab", enableExtensions && useMC ? 1 : 0, ""},
    BranchDefinition<std::vector<o2::its::TrackITS>>{InputSpec{"itsshorttracks", "GLO", "ITSSHORTTRACKS", 0}, "ITSShortTracks", enableShortTracks ? 1 : 0, ""},
    BranchDefinition<std::vector<int>>{InputSpec{"itsshortpv", "GLO", "ITSSHORTPV", 0}, "ITSShortTrackPVRefs", enableShortTracks ? 1 : 0, ""},
    BranchDefinition<std::vector<int>>{InputSpec{"itsshortclref", "GLO", "ITSSHORTCLREF", 0}, "ITSShortTrackClusRefs", enableShortTracks ? 1 : 0, ""},
    BranchDefinition<LabelsType>{InputSpec{"itsshort_mc", "GLO", "ITSSHORTMC", 0}, "ITSShortTrackMCLab", enableShortTracks && useMC ? 1 : 0, ""},
    BranchDefinition<o2::dataformats::StrangenessTrackingStats>{InputSpec{"strtrackingstats", "GLO", "STRTRACKSTATS", 0}, "StrangenessTrackingStats", enableExtensions || enableShortTracks ? 1 : 0, ""})();
}
} // namespace o2::strangeness_tracking
