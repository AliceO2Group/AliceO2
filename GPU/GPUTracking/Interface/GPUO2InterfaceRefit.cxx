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

/// \file GPUO2InterfaceRefit.cxx
/// \author David Rohr

#include "GPUO2InterfaceRefit.h"
#include "GPUO2InterfaceUtils.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "GPUParam.h"
#include "GPUTPCGMMergedTrackHit.h"
#include "GPUTrackingRefit.h"
#include "CorrectionMapsHelper.h"
#include "GPUTPCClusterOccupancyMap.h"

using namespace o2::gpu;
using namespace o2::tpc;

void GPUO2InterfaceRefit::fillSharedClustersAndOccupancyMap(const ClusterNativeAccess* cl, const gsl::span<const TrackTPC> trks, const TPCClRefElem* trackRef, uint8_t* shmap, uint32_t* ocmap, uint32_t nHbfPerTf, const GPUParam* param)
{
  if (!cl || (!shmap && cl->nClustersTotal > 0)) {
    throw std::runtime_error("Must provide clusters access and preallocated buffer for shared map");
  }
  std::unique_ptr<GPUParam> tmpParam;
  if (param == nullptr) {
    tmpParam = GPUO2InterfaceUtils::getFullParam(0.f, nHbfPerTf);
    param = tmpParam.get();
  }
  if ((param->rec.tpc.occupancyMapTimeBins || param->rec.tpc.sysClusErrorC12Norm) && ocmap && !nHbfPerTf) {
    throw std::runtime_error("Must provide nHbfPerTf for occupancy map");
  }
  memset(shmap, 0, sizeof(char) * cl->nClustersTotal);
  for (uint32_t i = 0; i < trks.size(); i++) {
    for (int32_t j = 0; j < trks[i].getNClusterReferences(); j++) {
      size_t idx = &trks[i].getCluster(trackRef, j, *cl) - cl->clustersLinear;
      shmap[idx] = shmap[idx] ? 2 : 1;
    }
  }
  std::vector<uint32_t> tmp;
  uint32_t* binmap = nullptr;
  if (ocmap && nHbfPerTf) {
    tmp.resize(param->rec.tpc.occupancyMapTimeBinsAverage ? GPUTPCClusterOccupancyMapBin::getNBins(*param) : 0, 0);
    binmap = param->rec.tpc.occupancyMapTimeBinsAverage ? tmp.data() : (ocmap + 2);
    *ocmap = cl->nClustersTotal / nHbfPerTf;
    if (param->rec.tpc.occupancyMapTimeBins) {
      ocmap[1] = param->rec.tpc.occupancyMapTimeBins * 0x10000 + param->rec.tpc.occupancyMapTimeBinsAverage;
    }
  }

  for (uint32_t i = 0; i < cl->nClustersTotal; i++) {
    shmap[i] = (shmap[i] > 1 ? GPUTPCGMMergedTrackHit::flagShared : 0) | cl->clustersLinear[i].getFlags();
    if (binmap) {
      binmap[(uint32_t)(cl->clustersLinear[i].getTime() / param->rec.tpc.occupancyMapTimeBins)]++;
    }
  }

  if (ocmap && nHbfPerTf && param->rec.tpc.occupancyMapTimeBinsAverage) {
    for (uint32_t bin = 0; bin < GPUTPCClusterOccupancyMapBin::getNBins(*param); bin++) {
      int32_t binmin = CAMath::Max<int32_t>(0, bin - param->rec.tpc.occupancyMapTimeBinsAverage);
      int32_t binmax = CAMath::Min<int32_t>(GPUTPCClusterOccupancyMapBin::getNBins(*param), bin + param->rec.tpc.occupancyMapTimeBinsAverage + 1);
      uint32_t sum = 0;
      for (int32_t i = binmin; i < binmax; i++) {
        sum += binmap[i];
      }
      sum /= binmax - binmin;
      ocmap[2 + bin] = sum;
    }
  }
}

size_t GPUO2InterfaceRefit::fillOccupancyMapGetSize(uint32_t nHbfPerTf, const GPUParam* param)
{
  std::unique_ptr<GPUParam> tmpParam;
  if (param == nullptr) {
    tmpParam = GPUO2InterfaceUtils::getFullParam(0.f, nHbfPerTf);
    param = tmpParam.get();
  }
  if ((param->rec.tpc.occupancyMapTimeBins || param->rec.tpc.sysClusErrorC12Norm) && !nHbfPerTf) {
    throw std::runtime_error("nHbfPerTf must not be zero for creation of the occupancy map");
  }
  if (param->rec.tpc.occupancyMapTimeBins) {
    return (GPUTPCClusterOccupancyMapBin::getNBins(*param) + 2) * sizeof(uint32_t);
  } else if (param->rec.tpc.sysClusErrorC12Norm) {
    return sizeof(uint32_t);
  } else {
    return 0;
  }
}

GPUO2InterfaceRefit::GPUO2InterfaceRefit(const ClusterNativeAccess* cl, const CorrectionMapsHelper* trans, float bzNominalGPU, const TPCClRefElem* trackRef, uint32_t nHbfPerTf, const uint8_t* sharedmap, const uint32_t* occupancymap, int32_t occupancyMapSize, const std::vector<TrackTPC>* trks, o2::base::Propagator* p)
{
  mParam = GPUO2InterfaceUtils::getFullParam(bzNominalGPU, nHbfPerTf);
  size_t expectedOccMapSize = nHbfPerTf ? fillOccupancyMapGetSize(nHbfPerTf, mParam.get()) : 0;
  if (cl->nClustersTotal) {
    if (sharedmap == nullptr && trks == nullptr) {
      throw std::runtime_error("Must provide either shared cluster map or vector of tpc tracks to build the map");
    }
    if ((sharedmap == nullptr) ^ (expectedOccMapSize && occupancymap == nullptr)) {
      throw std::runtime_error("Must provide either both shared cluster map and occupancy map or none of them");
    }
    if (sharedmap == nullptr) {
      mSharedMap.resize(cl->nClustersTotal);
      sharedmap = mSharedMap.data();
      mOccupancyMap.resize(expectedOccMapSize / sizeof(*mOccupancyMap.data()));
      occupancymap = mOccupancyMap.data();
      occupancyMapSize = expectedOccMapSize;
      fillSharedClustersAndOccupancyMap(cl, *trks, trackRef, mSharedMap.data(), mOccupancyMap.data(), nHbfPerTf, mParam.get());
    }
  }
  GPUO2InterfaceUtils::paramUseExternalOccupancyMap(mParam.get(), nHbfPerTf, occupancymap, occupancyMapSize);

  mRefit = std::make_unique<GPUTrackingRefit>();
  mRefit->SetGPUParam(mParam.get());
  mRefit->SetClusterStateArray(sharedmap);
  mRefit->SetPropagator(p);
  mRefit->SetClusterNative(cl);
  mRefit->SetTrackHitReferences(trackRef);
  mRefit->SetFastTransformHelper(trans);
}

void GPUO2InterfaceRefit::updateCalib(const CorrectionMapsHelper* trans, float bzNominalGPU)
{
  mParam->UpdateBzOnly(bzNominalGPU);
  mRefit->SetFastTransformHelper(trans);
}

int32_t GPUO2InterfaceRefit::RefitTrackAsGPU(o2::tpc::TrackTPC& trk, bool outward, bool resetCov) { return mRefit->RefitTrackAsGPU(trk, outward, resetCov); }
int32_t GPUO2InterfaceRefit::RefitTrackAsTrackParCov(o2::tpc::TrackTPC& trk, bool outward, bool resetCov) { return mRefit->RefitTrackAsTrackParCov(trk, outward, resetCov); }
int32_t GPUO2InterfaceRefit::RefitTrackAsGPU(o2::track::TrackParCov& trk, const o2::tpc::TrackTPCClusRef& clusRef, float time0, float* chi2, bool outward, bool resetCov) { return mRefit->RefitTrackAsGPU(trk, clusRef, time0, chi2, outward, resetCov); }
int32_t GPUO2InterfaceRefit::RefitTrackAsTrackParCov(o2::track::TrackParCov& trk, const o2::tpc::TrackTPCClusRef& clusRef, float time0, float* chi2, bool outward, bool resetCov) { return mRefit->RefitTrackAsTrackParCov(trk, clusRef, time0, chi2, outward, resetCov); }
void GPUO2InterfaceRefit::setIgnoreErrorsAtTrackEnds(bool v) { mRefit->mIgnoreErrorsOnTrackEnds = v; }
void GPUO2InterfaceRefit::setTrackReferenceX(float v) { mParam->rec.tpc.trackReferenceX = v; }

GPUO2InterfaceRefit::~GPUO2InterfaceRefit() = default;
