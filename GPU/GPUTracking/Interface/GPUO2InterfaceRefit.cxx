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
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "GPUParam.h"
#include "GPUTPCGMMergedTrackHit.h"
#include "GPUTrackingRefit.h"

using namespace o2::gpu;
using namespace o2::tpc;

void GPUO2InterfaceRefit::fillSharedClustersMap(const ClusterNativeAccess* cl, const gsl::span<const TrackTPC> trks, const TPCClRefElem* trackRef, unsigned char* shmap)
{
  if (!cl || !shmap) {
    throw std::runtime_error("Must provide clusters access and preallocated recepient for shared map");
  }
  memset(shmap, 0, sizeof(char) * cl->nClustersTotal);
  for (unsigned int i = 0; i < trks.size(); i++) {
    for (int j = 0; j < trks[i].getNClusterReferences(); j++) {
      size_t idx = &trks[i].getCluster(trackRef, j, *cl) - cl->clustersLinear;
      shmap[idx] = shmap[idx] ? 2 : 1;
    }
  }
  for (unsigned int i = 0; i < cl->nClustersTotal; i++) {
    shmap[i] = (shmap[i] > 1 ? GPUTPCGMMergedTrackHit::flagShared : 0) | cl->clustersLinear[i].getFlags();
  }
}

GPUO2InterfaceRefit::GPUO2InterfaceRefit(const ClusterNativeAccess* cl, const TPCFastTransform* trans, float bz, const TPCClRefElem* trackRef, const unsigned char* sharedmap, const std::vector<TrackTPC>* trks, o2::base::Propagator* p) : mParam(new GPUParam)
{
  if (cl->nClustersTotal) {
    if (sharedmap == nullptr && trks == nullptr) {
      throw std::runtime_error("Must provide either shared cluster map or vector of tpc tracks to build the map");
    }
    if (sharedmap == nullptr) {
      mSharedMap.resize(cl->nClustersTotal);
      sharedmap = mSharedMap.data();
      fillSharedClustersMap(cl, *trks, trackRef, mSharedMap.data());
    }
  }
  mRefit = std::make_unique<GPUTrackingRefit>();
  mParam->SetDefaults(bz);
  mRefit->SetGPUParam(mParam.get());
  mRefit->SetClusterStateArray(sharedmap);
  mRefit->SetPropagator(p);
  mRefit->SetClusterNative(cl);
  mRefit->SetTrackHitReferences(trackRef);
  mRefit->SetFastTransform(trans);
}

void GPUO2InterfaceRefit::updateCalib(const TPCFastTransform* trans, float bz)
{
  mParam->SetDefaults(bz);
  mRefit->SetGPUParam(mParam.get());
  mRefit->SetFastTransform(trans);
}

int GPUO2InterfaceRefit::RefitTrackAsGPU(o2::tpc::TrackTPC& trk, bool outward, bool resetCov) { return mRefit->RefitTrackAsGPU(trk, outward, resetCov); }
int GPUO2InterfaceRefit::RefitTrackAsTrackParCov(o2::tpc::TrackTPC& trk, bool outward, bool resetCov) { return mRefit->RefitTrackAsTrackParCov(trk, outward, resetCov); }
int GPUO2InterfaceRefit::RefitTrackAsGPU(o2::track::TrackParCov& trk, const o2::tpc::TrackTPCClusRef& clusRef, float time0, float* chi2, bool outward, bool resetCov) { return mRefit->RefitTrackAsGPU(trk, clusRef, time0, chi2, outward, resetCov); }
int GPUO2InterfaceRefit::RefitTrackAsTrackParCov(o2::track::TrackParCov& trk, const o2::tpc::TrackTPCClusRef& clusRef, float time0, float* chi2, bool outward, bool resetCov) { return mRefit->RefitTrackAsTrackParCov(trk, clusRef, time0, chi2, outward, resetCov); }
void GPUO2InterfaceRefit::setIgnoreErrorsAtTrackEnds(bool v) { mRefit->mIgnoreErrorsOnTrackEnds = v; }
void GPUO2InterfaceRefit::setGPUTrackFitInProjections(bool v) { mParam->rec.fitInProjections = v; }
void GPUO2InterfaceRefit::setTrackReferenceX(float v) { mParam->rec.tpc.trackReferenceX = v; }

GPUO2InterfaceRefit::~GPUO2InterfaceRefit() = default;
