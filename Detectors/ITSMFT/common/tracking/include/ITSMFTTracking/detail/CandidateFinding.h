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

#ifndef ALICEO2_ITSMFT_TRACKING_CANDIDATEFINDING_H_
#define ALICEO2_ITSMFT_TRACKING_CANDIDATEFINDING_H_

#ifndef GPUCA_GPUCODE
#include "ITSMFTTracking/GlobalMeasurement.h"
#include "ITSMFTTracking/IndexTableUtils.h"
#include "ITSMFTTracking/SurfaceDescriptor.h"
#include "ITSMFTTracking/TrackingPrimitives.h"
#endif

#ifndef GPUCA_GPUCODE
namespace o2::dataformats
{
template <typename Stamp>
class Vertex;
}
namespace o2::its
{
class TimeEstBC;
using Vertex = o2::dataformats::Vertex<TimeEstBC>;
} // namespace o2::its
#endif

namespace o2::itsmft::tracking
{

#ifndef GPUCA_GPUCODE

struct TrackletProjectionCache {
  int fromLayer;
  int toLayer;
  float fromRadius;
  float toRadius;
  float targetMinR;
  float targetMaxR;
  float targetMinZ;
  float targetMaxZ;
  float sourcePositionResolution;
  float edgeMSAngle;
  float edgePhiCut;
};

struct TrackletSearchWindow {
  int4 bins;
  float sourceReferenceCoordinate{0.f};
  float sourceProjectedCoordinate{0.f};
  float slope{0.f};
  float varianceConstant{0.f};
  float varianceLinear{0.f};
  float varianceQuadratic{0.f};
  float phiPrediction{0.f};
  float phiVariance{0.f};
};

bool projectTrackletSearchWindow(const GlobalMeasurement& sourceMeasurement,
                                 const o2::its::Vertex& vertex,
                                 float beamPositionVariance,
                                 SurfaceKind kind,
                                 const TrackletProjectionCache& edgeCache,
                                 const o2::itsmft::IndexTableUtilsCore& indexUtils,
                                 float nSigmaCut,
                                 TrackletSearchWindow& out);

#endif

} // namespace o2::itsmft::tracking

#endif
