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

/// \file GPUTRDO2BaseTrack.h
/// \author Ole Schmidt

#ifndef GPUTRDO2BASETRACK_H
#define GPUTRDO2BASETRACK_H

#include "GPUCommonDef.h"
#include "ReconstructionDataFormats/Track.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class GPUTRDO2BaseTrack : public o2::track::TrackParCov
{
 public:
  GPUdDefault() GPUTRDO2BaseTrack() = default;
  GPUd() GPUTRDO2BaseTrack(const o2::track::TrackParCov& t) : o2::track::TrackParCov(t) {}

 private:
  // dummy class to avoid problems, see https://github.com/AliceO2Group/AliceO2/pull/5969#issuecomment-827475822
  ClassDefNV(GPUTRDO2BaseTrack, 1);
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
