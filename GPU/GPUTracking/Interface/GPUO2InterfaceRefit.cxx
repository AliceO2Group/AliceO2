// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

using namespace o2::gpu;
using namespace o2::tpc;

GPUTPCO2InterfaceRefit::GPUTPCO2InterfaceRefit(const o2::tpc::ClusterNativeAccess* cl, const TPCFastTransform* trans, float bz, const TPCClRefElem* trackRef, const unsigned char* sharedmap, std::vector<o2::tpc::TrackTPC>* trks, o2::base::Propagator* p) : mRefit(), mParam(new GPUParam)
{
  if (sharedmap == nullptr && trks == nullptr) {
    throw std::runtime_error("Must provide either shared cluster map or vector of tpc tracks to build the map");
  }
  if (sharedmap == nullptr) {
    mSharedMap.resize(cl->nClustersTotal);
    sharedmap = mSharedMap.data();
    std::fill(mSharedMap.begin(), mSharedMap.end(), 0);
    for (unsigned int i = 0; i < (*trks).size(); i++) {
      for (unsigned int j = 0; j < (*trks)[i].getNClusterReferences(); j++) {
        size_t idx = &(*trks)[i].getCluster(trackRef, j, *cl) - cl->clustersLinear;
        mSharedMap[idx] = mSharedMap[idx] ? 2 : 1;
      }
    }
    for (unsigned int i = 0; i < cl->nClustersTotal; i++) {
      mSharedMap[i] = (mSharedMap[i] > 1 ? GPUTPCGMMergedTrackHit::flagShared : 0) | cl->clustersLinear[i].getFlags();
    }
  }

  mParam->SetDefaults(bz);
  mRefit.SetGPUParam(mParam.get());
  mRefit.SetClusterStateArray(sharedmap);
  mRefit.SetPropagator(p);
  mRefit.SetClusterNative(cl);
  mRefit.SetTrackHitReferences(trackRef);
  mRefit.SetFastTransform(trans);
}

void GPUTPCO2InterfaceRefit::setGPUTrackFitInProjections(bool v) { mParam->rec.fitInProjections = v; }
void GPUTPCO2InterfaceRefit::setTrackReferenceX(float v) { mParam->rec.TrackReferenceX = v; }

GPUTPCO2InterfaceRefit::~GPUTPCO2InterfaceRefit() = default;
