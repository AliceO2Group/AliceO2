// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUO2InterfaceRefit.h
/// \author David Rohr

#ifndef GPUO2INTERFACEREFIT_H
#define GPUO2INTERFACEREFIT_H

// Some defines denoting that we are compiling for O2
#ifndef HAVE_O2HEADERS
#define HAVE_O2HEADERS
#endif
#ifndef GPUCA_TPC_GEOMETRY_O2
#define GPUCA_TPC_GEOMETRY_O2
#endif
#ifndef GPUCA_O2_INTERFACE
#define GPUCA_O2_INTERFACE
#endif

#include "GPUTrackingRefit.h"
#include <memory>
#include <vector>
#include <gsl/span>

namespace o2::tpc
{
using TPCClRefElem = uint32_t;
}

namespace o2::gpu
{
class GPUParam;
class GPUO2InterfaceRefit
{
 public:
  // Must initialize with:
  // - In any case: Cluster Native access structure (cl), TPC Fast Transformation instance (trans), solenoid field (bz), TPC Track hit references (trackRef)
  // - Either the shared cluster map (sharedmap) or the vector of tpc tracks (trks) to build the shared cluster map internally
  // - o2::base::Propagator (p) in case RefitTrackAsTrackParCov is to be used

  GPUO2InterfaceRefit(const o2::tpc::ClusterNativeAccess* cl, const TPCFastTransform* trans, float bz, const o2::tpc::TPCClRefElem* trackRef, const unsigned char* sharedmap = nullptr, const std::vector<o2::tpc::TrackTPC>* trks = nullptr, o2::base::Propagator* p = nullptr);
  ~GPUO2InterfaceRefit();

  int RefitTrackAsGPU(o2::tpc::TrackTPC& trk, bool outward = false, bool resetCov = false) { return mRefit.RefitTrackAsGPU(trk, outward, resetCov); }
  int RefitTrackAsTrackParCov(o2::tpc::TrackTPC& trk, bool outward = false, bool resetCov = false) { return mRefit.RefitTrackAsTrackParCov(trk, outward, resetCov); }
  int RefitTrackAsGPU(o2::track::TrackParCov& trk, const o2::tpc::TrackTPCClusRef& clusRef, float time0, float* chi2 = nullptr, bool outward = false, bool resetCov = false) { return mRefit.RefitTrackAsGPU(trk, clusRef, time0, chi2, outward, resetCov); }
  int RefitTrackAsTrackParCov(o2::track::TrackParCov& trk, const o2::tpc::TrackTPCClusRef& clusRef, float time0, float* chi2 = nullptr, bool outward = false, bool resetCov = false) { return mRefit.RefitTrackAsTrackParCov(trk, clusRef, time0, chi2, outward, resetCov); }
  void setGPUTrackFitInProjections(bool v = true);
  void setTrackReferenceX(float v);
  void setIgnoreErrorsAtTrackEnds(bool v) { mRefit.mIgnoreErrorsOnTrackEnds = v; }

  static void fillSharedClustersMap(const o2::tpc::ClusterNativeAccess* cl, const gsl::span<const o2::tpc::TrackTPC> trks, const o2::tpc::TPCClRefElem* trackRef, unsigned char* shmap);

 private:
  GPUTrackingRefit mRefit;
  std::unique_ptr<GPUParam> mParam;
  std::vector<unsigned char> mSharedMap;
};
} // namespace o2::gpu

#endif
