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

// Skeleton derived from RS's code in ITSOffStudy

#include "ITSStudies/ImpactParameter.h"

#include "Framework/CCDBParamSpec.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"

namespace o2
{
namespace its
{
namespace study
{
using namespace o2::framework;
using namespace o2::globaltracking;

using PVertex = o2::dataformats::PrimaryVertex;
using GTrackID = o2::dataformats::GlobalTrackID;

class ImpactParameterStudy : public Task
{
 public:
  ImpactParameterStudy(std::shared_ptr<DataRequest> dr, mask_t src) : mDataRequest(dr), mTracksSrc(src){};
  ~ImpactParameterStudy() final = default;
  void run(ProcessingContext&) final;
  void endOfStream(EndOfStreamContext&) final;
  void finaliseCCDB(ConcreteDataMatcher&, void*) final;
  void process(o2::globaltracking::RecoContainer&);

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  GTrackID::mask_t mTracksSrc{};

  // Data
  std::shared_ptr<DataRequest> mDataRequest;
  gsl::span<const PVertex> mPVertices;
};

void ImpactParameterStudy::run(ProcessingContext& pc)
{
  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());
  updateTimeDependentParams(pc); // Make sure this is called after recoData.collectData, which may load some conditions
  process(recoData);
}

void ImpactParameterStudy::process(o2::globaltracking::RecoContainer& recoData)
{
  auto trackIndex = recoData.getPrimaryVertexMatchedTracks(); // Global ID's for associated tracks
  auto vtxRefs = recoData.getPrimaryVertexMatchedTrackRefs(); // references from vertex to these track IDs
  auto pvertices = recoData.getPrimaryVertices();

  int nv = vtxRefs.size() - 1;      // The last entry is for unassigned tracks, ignore them
  for (int iv = 0; iv < nv; iv++) { // Loop over PVs
    const auto& vtref = vtxRefs[iv];
    const auto& pv = pvertices[iv];
    int it = vtref.getFirstEntry(), itLim = it + vtref.getEntries();
    pv.print();
    for (; it < itLim; it++) {
      auto tvid = trackIndex[it];
      if (!recoData.isTrackSourceLoaded(tvid.getSource())) {
        continue;
      }
      const auto& trc = recoData.getTrackParam(tvid); // The actual track
      if (it < 5) {
        trc.print();
      }
    }
  }
}

void ImpactParameterStudy::updateTimeDependentParams(ProcessingContext& pc)
{
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
  }
}

void ImpactParameterStudy::endOfStream(EndOfStreamContext& ec)
{
}

void ImpactParameterStudy::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
}

DataProcessorSpec getImpactParameterStudy(mask_t srcTracksMask, mask_t srcClustersMask, bool useMC)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();
  dataRequest->requestTracks(srcTracksMask, useMC);
  // dataRequest->requestClusters(srcClustersMask, useMC);
  dataRequest->requestPrimaryVertertices(useMC);

  return DataProcessorSpec{
    "its-study-impactparameter",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<ImpactParameterStudy>(dataRequest, srcTracksMask)},
    Options{}};
}
} // namespace study
} // namespace its
} // namespace o2