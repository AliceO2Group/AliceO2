// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTrackingRefit.h
/// \author David Rohr

#ifndef GPUTRACKINGREFIT_H
#define GPUTRACKINGREFIT_H

#include "GPUDef.h"
#include "GPUProcessor.h"

namespace o2::dataformats
{
template <typename FirstEntry, typename NElem>
class RangeReference;
}
namespace o2::track
{
template <typename T>
class TrackParametrizationWithError;
using TrackParCov = TrackParametrizationWithError<float>;
} // namespace o2::track
namespace o2::base
{
class Propagator;
class MatLayerCylSet;
} // namespace o2::base
namespace o2::tpc
{
struct ClusterNativeAccess;
class TrackTPC;
using TrackTPCClusRef = o2::dataformats::RangeReference<uint32_t, uint16_t>;
} // namespace o2::tpc

namespace o2::gpu
{
class GPUTPCGMTrackParam;
class GPUTPCGMMergedTrack;
MEM_CLASS_PRE()
struct GPUConstantMem;
MEM_CLASS_PRE()
struct GPUParam;
struct GPUTPCGMMergedTrackHit;
class TPCFastTransform;

class GPUTrackingRefit
{
 public:
  void SetClusterStateArray(const unsigned char* v) { mPclusterState = v; }
  void SetPtrsFromGPUConstantMem(const GPUConstantMem* v, MEM_CONSTANT(GPUParam) * p = nullptr);
  void SetPropagator(const o2::base::Propagator* v) { mPpropagator = v; }
  void SetPropagatorDefault();
  void SetClusterNative(const o2::tpc::ClusterNativeAccess* v) { mPclusterNative = v; }
  void SetTrackHits(const GPUTPCGMMergedTrackHit* v) { mPtrackHits = v; }
  void SetTrackHitReferences(const unsigned int* v) { mPtrackHitReferences = v; }
  void SetFastTransform(const TPCFastTransform* v) { mPfastTransform = v; }
  void SetGPUParam(const MEM_CONSTANT(GPUParam) * v) { mPparam = v; }
  GPUd() int RefitTrackAsGPU(GPUTPCGMMergedTrack& trk, bool outward = false, bool resetCov = false) { return RefitTrack<GPUTPCGMMergedTrack, GPUTPCGMTrackParam>(trk, outward, resetCov); }
  GPUd() int RefitTrackAsTrackParCov(GPUTPCGMMergedTrack& trk, bool outward = false, bool resetCov = false) { return RefitTrack<GPUTPCGMMergedTrack, o2::track::TrackParCov>(trk, outward, resetCov); }
  GPUd() int RefitTrackAsGPU(o2::tpc::TrackTPC& trk, bool outward = false, bool resetCov = false) { return RefitTrack<o2::tpc::TrackTPC, GPUTPCGMTrackParam>(trk, outward, resetCov); }
  GPUd() int RefitTrackAsTrackParCov(o2::tpc::TrackTPC& trk, bool outward = false, bool resetCov = false) { return RefitTrack<o2::tpc::TrackTPC, o2::track::TrackParCov>(trk, outward, resetCov); }

  struct TrackParCovWithArgs {
    o2::track::TrackParCov& trk;
    const o2::tpc::TrackTPCClusRef& clusRef;
    float time0;
    float* chi2;
  };
  GPUd() int RefitTrackAsGPU(o2::track::TrackParCov& trk, const o2::tpc::TrackTPCClusRef& clusRef, float time0, float* chi2 = nullptr, bool outward = false, bool resetCov = false)
  {
    TrackParCovWithArgs x{trk, clusRef, time0, chi2};
    return RefitTrack<TrackParCovWithArgs, GPUTPCGMTrackParam>(x, outward, resetCov);
  }
  GPUd() int RefitTrackAsTrackParCov(o2::track::TrackParCov& trk, const o2::tpc::TrackTPCClusRef& clusRef, float time0, float* chi2 = nullptr, bool outward = false, bool resetCov = false)
  {
    TrackParCovWithArgs x{trk, clusRef, time0, chi2};
    return RefitTrack<TrackParCovWithArgs, o2::track::TrackParCov>(x, outward, resetCov);
  }

  bool mIgnoreErrorsOnTrackEnds = true; // Ignore errors during propagation / update at the beginning / end of tracks for short tracks / tracks with high incl. angle

 private:
  const unsigned char* mPclusterState = nullptr;                 // Ptr to shared cluster state
  const o2::base::Propagator* mPpropagator = nullptr;            // Ptr to propagator for TrackParCov track model
  const o2::base::MatLayerCylSet* mPmatLUT = nullptr;            // Ptr to material LUT
  const o2::tpc::ClusterNativeAccess* mPclusterNative = nullptr; // Ptr to cluster native access structure
  const GPUTPCGMMergedTrackHit* mPtrackHits = nullptr;           // Ptr to hits for GPUTPCGMMergedTrack tracks
  const unsigned int* mPtrackHitReferences = nullptr;            // Ptr to hits for TrackTPC tracks
  const TPCFastTransform* mPfastTransform = nullptr;             // Ptr to TPC fast transform object
  const MEM_CONSTANT(GPUParam) * mPparam = nullptr;              // Ptr to GPUParam
  template <class T, class S>
  GPUd() int RefitTrack(T& trk, bool outward, bool resetCov);
  template <class T, class S, class U>
  GPUd() void convertTrack(T& trk, const S& trkX, U& prop, float* chi2);
  template <class U>
  GPUd() void initProp(U& prop);
};

class GPUTrackingRefitProcessor : public GPUTrackingRefit, public GPUProcessor
{
 public:
#ifndef GPUCA_GPUCODE
  void InitializeProcessor();
  void RegisterMemoryAllocation();
  void SetMaxData(const GPUTrackingInOutPointers& io);
#endif
  GPUTPCGMMergedTrack* mPTracks = nullptr;
};

} // namespace o2::gpu

#endif
