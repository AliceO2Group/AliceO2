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

/// \file TrackWriterSpec.cxx
/// \brief Implementation of a data processor to write matched MCH-MID tracks in a root file
///
/// \author Philippe Pillot, Subatech

#include "TrackWriterSpec.h"

#include <vector>

#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "ReconstructionDataFormats/TrackMCHMID.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{
namespace muon
{

using namespace o2::framework;

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

DataProcessorSpec getTrackWriterSpec(bool useMC, const char* specName, const char* fileName)
{
  return MakeRootTreeWriterSpec(specName,
                                fileName,
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree Matched MCH-MID Tracks"},
                                BranchDefinition<std::vector<dataformats::TrackMCHMID>>{InputSpec{"tracks", "GLO", "MTC_MCHMID"}, "tracks"},
                                BranchDefinition<std::vector<o2::MCCompLabel>>{InputSpec{"tracklabels", "GLO", "MCMTC_MCHMID"}, "tracklabels", useMC ? 1 : 0})();
}

} // namespace muon
} // namespace o2
