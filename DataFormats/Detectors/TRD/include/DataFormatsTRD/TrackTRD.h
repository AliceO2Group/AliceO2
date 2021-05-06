// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
} // namespace o2

#endif // O2_DATAFORMATS_TRACK_TRD_H
