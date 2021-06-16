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

/// \file TrackTRD.h
/// \author David Rohr

#ifndef O2_DATAFORMATS_TRACK_TRD_H
#define O2_DATAFORMATS_TRACK_TRD_H

#include "GPUTRDTrack.h"

namespace o2
{
namespace trd
{
using TrackTRD = o2::gpu::GPUTRDTrack;
} // namespace trd
namespace framework
{
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::trd::TrackTRD> : std::true_type {
};
} // namespace framework
namespace gpu
{
static_assert(sizeof(o2::dataformats::GlobalTrackID) == sizeof(unsigned int));
template <>
GPUdi() o2::dataformats::GlobalTrackID GPUTRDTrack_t<trackInterface<GPUTRDO2BaseTrack>>::getRefGlobalTrackId() const
{
  return o2::dataformats::GlobalTrackID{mRefGlobalTrackId};
}
template <>
GPUdi() void GPUTRDTrack_t<trackInterface<GPUTRDO2BaseTrack>>::setRefGlobalTrackId(o2::dataformats::GlobalTrackID id)
{
  setRefGlobalTrackIdRaw(id.getRaw());
}
} // namespace gpu
} // namespace o2

#endif // O2_DATAFORMATS_TRACK_TRD_H
