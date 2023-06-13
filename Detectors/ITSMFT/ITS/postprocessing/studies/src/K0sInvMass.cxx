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

#include "ITSStudies/K0sInvMass.h"

#include "Framework/CCDBParamSpec.h"
#include "Framework/Task.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"

namespace o2
{
namespace its
{
namespace study
{
using namespace o2::framework;
using namespace o2::globaltracking;

using GTrackID = o2::dataformats::GlobalTrackID;

class K0InvMassStudy : public Task
{
 public:
  K0InvMassStudy(std::shared_ptr<DataRequest> dr, mask_t src) : mDataRequest(dr), mTracksSrc(src){};
  ~K0InvMassStudy() final = default;
  void run(ProcessingContext&) final;
  void endOfStream(EndOfStreamContext&) final;
  void finaliseCCDB(ConcreteDataMatcher&, void*) final;
  void process(o2::globaltracking::RecoContainer&);

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  GTrackID::mask_t mTracksSrc{};

  // Data
  std::shared_ptr<DataRequest> mDataRequest;
};

void K0InvMassStudy::run(ProcessingContext& pc)
{
  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());
  updateTimeDependentParams(pc); // Make sure this is called after recoData.collectData, which may load some conditions
  process(recoData);
}

void K0InvMassStudy::process(o2::globaltracking::RecoContainer& recoData)
{
  auto v0 = recoData.getV0s();
  LOGP(info, " ====> got {} v0s", v0.size());
}

void K0InvMassStudy::updateTimeDependentParams(ProcessingContext& pc)
{
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
  }
}

void K0InvMassStudy::endOfStream(EndOfStreamContext& ec)
{
}

void K0InvMassStudy::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
}

DataProcessorSpec getK0sInvMassStudy(mask_t srcTracksMask, bool useMC)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();
  dataRequest->requestTracks(srcTracksMask, useMC);
  dataRequest->requestSecondaryVertices(useMC);

  return DataProcessorSpec{
    "its-study-k0sinvmass",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<K0InvMassStudy>(dataRequest, srcTracksMask)},
    Options{}};
}
} // namespace study
} // namespace its
} // namespace o2