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

#include "ITSStudies/ITSZDCAnomalyStudy.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "DataFormatsParameters/GRPObject.h"

#include "Framework/Task.h"
#include "Framework/Logger.h"

using namespace o2::framework;
using namespace o2::globaltracking;
using GTrackID = o2::dataformats::GlobalTrackID;

namespace o2::its::study
{
class ITSZDCAnomalyStudy : public Task
{
 public:
  ITSZDCAnomalyStudy(std::shared_ptr<DataRequest> dr,
                     std::shared_ptr<o2::base::GRPGeomRequest> gr,
                     bool isMC) : mDataRequest{dr}, mGGCCDBRequest(gr), mUseMC(isMC) {}

  void init(InitContext& ic) final;
  void run(ProcessingContext&) final;
  void endOfStream(EndOfStreamContext&) final;
  void finaliseCCDB(ConcreteDataMatcher&, void*) final;

 private:
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  std::shared_ptr<DataRequest> mDataRequest;
  bool mUseMC;
};

void ITSZDCAnomalyStudy::init(InitContext& ic)
{
  LOGP(info, "Initializing ITSZDCAnomalyStudy");
}

void ITSZDCAnomalyStudy::endOfStream(EndOfStreamContext&)
{
  LOGP(info, "End of stream for ITSZDCAnomalyStudy");
}

void ITSZDCAnomalyStudy::run(ProcessingContext& pc)
{
  LOGP(info, "Running ITSZDCAnomalyStudy");
}

void ITSZDCAnomalyStudy::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
//   if (matcher == ConcreteDataMatcher("ITS", "CLUSDICT", 0)) {
//     setClusterDictionary((const o2::itsmft::TopologyDictionary*)obj);
//     return;
//   }
}

// getter
DataProcessorSpec getITSZDCAnomalyStudy(mask_t srcClustersMask, bool useMC)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();
  dataRequest->requestClusters(srcClustersMask, useMC);
  dataRequest->requestTracks(GTrackID::getSourcesMask(""), useMC);

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              false,                             // GRPMagField
                                                              false,                             // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true);
  return DataProcessorSpec{
    "its-zdc-anomaly-study",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<ITSZDCAnomalyStudy>(dataRequest, ggRequest, useMC)},
    Options{}};
}

} // namespace o2::its::study