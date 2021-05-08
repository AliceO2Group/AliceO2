// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTRDTrackO2.cxx
/// \author David Rohr

#define GPU_TRD_TRACK_O2
#include "GPUTRDTrack.cxx"
#include "ReconstructionDataFormats/GlobalTrackID.h"

namespace o2::gpu
{
template class GPUTRDTrack_t<trackInterface<o2::gpu::GPUTRDO2BaseTrack>>;
} // namespace o2::gpu
