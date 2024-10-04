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

/// @file ITSTPCMacthingQCSpec.cxx

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "GlobalTrackingWorkflowQC/ITSTPCMatchingQCSpec.h"
#include "GLOQC/ITSTPCMatchingQCParams.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsVertexing/SVertexerParams.h"
#include "Framework/CCDBParamSpec.h"
#include "CommonUtils/NameConf.h"
#include <TFile.h>

using namespace o2::framework;

namespace o2
{
namespace globaltracking
{

void ITSTPCMatchingQCDevice::init(InitContext& /*ic*/)
{
  const o2::gloqc::ITSTPCMatchingQCParams& params = o2::gloqc::ITSTPCMatchingQCParams::Instance();

  mMatchITSTPCQC = std::make_unique<o2::gloqc::MatchITSTPCQC>();
  mMatchITSTPCQC->setDataRequest(mDataRequest);
  mMatchITSTPCQC->setTrkSources(o2::dataformats::GlobalTrackID::getSourcesMask(mTrkSources));
  mMatchITSTPCQC->setMinPtITSCut(params.minPtITSCut);
  mMatchITSTPCQC->setEtaITSCut(params.etaITSCut);
  mMatchITSTPCQC->setMinNClustersITS(params.minNITSClustersCut);
  mMatchITSTPCQC->setMaxChi2PerClusterITS(params.maxChi2PerClusterITS);
  mMatchITSTPCQC->setMinPtTPCCut(params.minPtTPCCut);
  mMatchITSTPCQC->setEtaTPCCut(params.etaTPCCut);
  mMatchITSTPCQC->setMinNTPCClustersCut(params.minNTPCClustersCut);
  mMatchITSTPCQC->setMinDCAtoBeamPipeDistanceCut(params.minDCACut);
  mMatchITSTPCQC->setMinDCAtoBeamPipeYCut(params.minDCACutY);
  mMatchITSTPCQC->setPtCut(params.minPtCut);
  mMatchITSTPCQC->setMaxPtCut(params.maxPtCut);
  mMatchITSTPCQC->setEtaCut(params.etaCut);
  mMatchITSTPCQC->setCutK0Mass(params.cutK0Mass);
  mMatchITSTPCQC->setMaxK0Eta(params.maxEtaK0);
  mMatchITSTPCQC->setK0Scaling(params.K0Scaling);
  o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
  if (mUseMC) {
    mMatchITSTPCQC->setUseMC(mUseMC);
  }
  if (mDoK0QC) {
    mMatchITSTPCQC->setDoK0QC(mDoK0QC);
  }
  mMatchITSTPCQC->init();
}

//_____________________________________________________________

void ITSTPCMatchingQCDevice::run(o2::framework::ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  static bool wasSVParamInitialized = false;
  if (!wasSVParamInitialized) {
    pc.inputs().get<o2::vertexing::SVertexerParams*>("SVParam");
    wasSVParamInitialized = true;
  }
  mMatchITSTPCQC->run(pc);
}

//_____________________________________________________________

void ITSTPCMatchingQCDevice::endOfStream(o2::framework::EndOfStreamContext& ec)
{

  mMatchITSTPCQC->finalize();
  sendOutput(ec.outputs());
}

//_____________________________________________________________

void ITSTPCMatchingQCDevice::sendOutput(DataAllocator& output)
{

  TObjArray objar;
  mMatchITSTPCQC->getHistos(objar);
  output.snapshot(Output{"GLO", "ITSTPCMATCHQC", 0}, objar);

  TFile* f = new TFile(Form("outITSTPCmatchingQC.root"), "RECREATE");
  if (f == nullptr) {
    LOGP(error, "Cannot write QC to file 'outITSTPCmatchingQC.root'");
    return;
  }
  objar.Write("ObjArray", TObject::kSingleKey);
  f->Close();
}

void ITSTPCMatchingQCDevice::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (matcher == ConcreteDataMatcher("GLO", "SVPARAM", 0)) {
    return;
  }
}
} // namespace globaltracking

namespace framework
{
using GID = o2::dataformats::GlobalTrackID;

DataProcessorSpec getITSTPCMatchingQCDevice(bool useMC, bool doK0QC, std::string trkSources)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("GLO", "ITSTPCMATCHQC", 0, Lifetime::Sporadic);

  auto dataRequest = std::make_shared<o2::globaltracking::DataRequest>();
  GID::mask_t srcMask = GID::getSourcesMask(trkSources);
  dataRequest->requestTracks(srcMask, useMC);
  if (doK0QC) {
    dataRequest->requestPrimaryVertices(useMC);
    dataRequest->requestSecondaryVertices(useMC);
    dataRequest->requestTPCClusters(false);
  }
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                          // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                true,                           // GRPLHCIF
                                                                true,                           // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                dataRequest->inputs);

  dataRequest->inputs.emplace_back("SVParam", "GLO", "SVPARAM", 0, Lifetime::Condition, ccdbParamSpec("GLO/Config/SVertexerParam"));
  return DataProcessorSpec{
    .name = "itstpc-matching-qc",
    .inputs = dataRequest->inputs,
    .outputs = outputs,
    .algorithm = AlgorithmSpec{adaptFromTask<o2::globaltracking::ITSTPCMatchingQCDevice>(dataRequest, ccdbRequest, useMC, doK0QC, trkSources)},
  };
}

} // namespace framework
} // namespace o2
