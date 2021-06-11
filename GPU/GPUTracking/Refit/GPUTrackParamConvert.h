// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTrackParamConvert.h
/// \author David Rohr

#ifndef O2_GPU_TRACKPARAMCONVERT_H
#define O2_GPU_TRACKPARAMCONVERT_H

#include "GPUO2DataTypes.h"
#include "GPUTPCGMTrackParam.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUTPCGMPropagator.h"
#include "ReconstructionDataFormats/Track.h"
#include "DetectorsBase/Propagator.h"
#include "DataFormatsTPC/TrackTPC.h"

namespace o2::gpu
{

GPUdi() static void convertTrackParam(GPUTPCGMTrackParam& trk, const o2::track::TrackParCov& trkX)
{
  for (int i = 0; i < 5; i++) {
    trk.Par()[i] = trkX.getParams()[i];
  }
  for (int i = 0; i < 15; i++) {
    trk.Cov()[i] = trkX.getCov()[i];
  }
  trk.X() = trkX.getX();
}
GPUdi() static void convertTrackParam(o2::track::TrackParCov& trk, const GPUTPCGMTrackParam& trkX)
{
  for (int i = 0; i < 5; i++) {
    trk.setParam(trkX.GetPar()[i], i);
  }
  for (int i = 0; i < 15; i++) {
    trk.setCov(trkX.GetCov()[i], i);
  }
  trk.setX(trkX.GetX());
}

} // namespace o2::gpu

#endif
