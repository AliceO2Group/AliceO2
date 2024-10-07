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

/// \file GPUO2InterfaceRefit.h
/// \author David Rohr

#ifndef GPUO2INTERFACEREFIT_H
#define GPUO2INTERFACEREFIT_H

// Some defines denoting that we are compiling for O2
#ifndef GPUCA_HAVE_O2HEADERS
#define GPUCA_HAVE_O2HEADERS
#endif
#ifndef GPUCA_TPC_GEOMETRY_O2
#define GPUCA_TPC_GEOMETRY_O2
#endif
#ifndef GPUCA_O2_INTERFACE
#define GPUCA_O2_INTERFACE
#endif

#include <memory>
#include <vector>
#include <gsl/span>

namespace o2::base
{
template <typename value_T>
class PropagatorImpl;
using Propagator = PropagatorImpl<float>;
} // namespace o2::base
namespace o2::dataformats
{
template <typename FirstEntry, typename NElem>
class RangeReference;
}
namespace o2::tpc
{
using TPCClRefElem = uint32_t;
using TrackTPCClusRef = o2::dataformats::RangeReference<uint32_t, uint16_t>;
class TrackTPC;
struct ClusterNativeAccess;
} // namespace o2::tpc
namespace o2::track
{
template <typename value_T>
class TrackParametrizationWithError;
using TrackParCovF = TrackParametrizationWithError<float>;
using TrackParCov = TrackParCovF;
} // namespace o2::track

namespace o2::gpu
{
class GPUParam;
class GPUTrackingRefit;
class CorrectionMapsHelper;
class GPUO2InterfaceRefit
{
 public:
  // Must initialize with:
  // - In any case: Cluster Native access structure (cl), TPC Fast Transformation helper instance (trans), solenoid field (bz), TPC Track hit references (trackRef)
  // - Either the shared cluster map (sharedmap) or the vector of tpc tracks (trks) to build the shared cluster map internally
  // - o2::base::Propagator (p) in case RefitTrackAsTrackParCov is to be used
  // - In case the --configKeyValues defining GPUParam settings require an occupancy map for TPC error estimation, the map must either be provided as occupancymap, or nHbfPerTf must be set non-zero

  GPUO2InterfaceRefit(const o2::tpc::ClusterNativeAccess* cl, const o2::gpu::CorrectionMapsHelper* trans, float bzNominalGPU, const o2::tpc::TPCClRefElem* trackRef, uint32_t nHbfPerTf = 0, const uint8_t* sharedmap = nullptr, const uint32_t* occupancymap = nullptr, int32_t occupancyMapSize = -1, const std::vector<o2::tpc::TrackTPC>* trks = nullptr, o2::base::Propagator* p = nullptr);
  ~GPUO2InterfaceRefit();

  int32_t RefitTrackAsGPU(o2::tpc::TrackTPC& trk, bool outward = false, bool resetCov = false);
  int32_t RefitTrackAsTrackParCov(o2::tpc::TrackTPC& trk, bool outward = false, bool resetCov = false);
  int32_t RefitTrackAsGPU(o2::track::TrackParCov& trk, const o2::tpc::TrackTPCClusRef& clusRef, float time0, float* chi2 = nullptr, bool outward = false, bool resetCov = false);
  int32_t RefitTrackAsTrackParCov(o2::track::TrackParCov& trk, const o2::tpc::TrackTPCClusRef& clusRef, float time0, float* chi2 = nullptr, bool outward = false, bool resetCov = false);
  void setTrackReferenceX(float v);
  void setIgnoreErrorsAtTrackEnds(bool v);
  void updateCalib(const o2::gpu::CorrectionMapsHelper* trans, float bzNominalGPU);
  auto getParam() const { return mParam.get(); }

  // To create shared cluster maps and occupancy maps.
  // param is an optional parameter to override the param object, by default a default object from --configKeyValues is used.
  // If the param object / default object requires an occupancy map, an occupancy map ptr and nHbfPerTf value must be provided.
  // You can use the function fillOccupancyMapGetSize(...) to get the required size of the occupancy map. If 0 is returned, no map is required.
  // Providing only the shmap ptr but no ocmap ptr will create only the shared map, but no occupancy map.
  static void fillSharedClustersAndOccupancyMap(const o2::tpc::ClusterNativeAccess* cl, const gsl::span<const o2::tpc::TrackTPC> trks, const o2::tpc::TPCClRefElem* trackRef, uint8_t* shmap, uint32_t* ocmap = nullptr, uint32_t nHbfPerTf = 0, const GPUParam* param = nullptr);
  static size_t fillOccupancyMapGetSize(uint32_t nHbfPerTf, const GPUParam* param = nullptr);

 private:
  std::unique_ptr<GPUTrackingRefit> mRefit;
  std::unique_ptr<GPUParam> mParam;
  std::vector<uint8_t> mSharedMap;
  std::vector<uint32_t> mOccupancyMap;
};
} // namespace o2::gpu

#endif
