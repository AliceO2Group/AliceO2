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

/// \file GPUTRDTrack.cxx
/// \author Ole Schmidt, Sergey Gorbunov

#include "GPUTRDTrack.h"
#include "GPUTRDInterfaces.h"

using namespace o2::gpu;
#include "GPUTRDTrack.inc"

#if !defined(GPUCA_GPUCODE)
namespace o2::gpu
{
#if defined(GPUCA_STANDALONE) // Instantiate O2 track version, for O2 this happens in GPUTRDTrackO2.cxx
template class GPUTRDTrack_t<trackInterface<o2::track::TrackParCov>>;
#endif
template class GPUTRDTrack_t<trackInterface<GPUTPCGMTrackParam>>; // Always instatiate GM track version
} // namespace o2::gpu
#endif
