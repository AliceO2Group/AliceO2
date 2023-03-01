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

#include "MCHIO/TrackWriterSpec.h"

#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "Framework/Logger.h"
#include "DataFormatsMCH/Cluster.h"
#include "DataFormatsMCH/Digit.h"
#include <vector>

using namespace o2::framework;

namespace o2::mch
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

DataProcessorSpec getTrackWriterSpec(bool useMC, const char* specName, const char* fileName, bool digits)
{
  return MakeRootTreeWriterSpec(specName,
                                fileName,
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree MCH Standalone Tracks"},
                                BranchDefinition<std::vector<TrackMCH>>{InputSpec{"tracks", "MCH", "TRACKS"}, "tracks"},
                                BranchDefinition<std::vector<ROFRecord>>{InputSpec{"trackrofs", "MCH", "TRACKROFS"}, "trackrofs"},
                                BranchDefinition<std::vector<Cluster>>{InputSpec{"trackclusters", "MCH", "TRACKCLUSTERS"}, "trackclusters"},
                                BranchDefinition<std::vector<Digit>>{InputSpec{"trackdigits", "MCH", "TRACKDIGITS"}, "trackdigits", digits ? 1 : 0},
                                BranchDefinition<std::vector<o2::MCCompLabel>>{InputSpec{"tracklabels", "MCH", "TRACKLABELS"}, "tracklabels", useMC ? 1 : 0})();
}

} // namespace o2::mch
