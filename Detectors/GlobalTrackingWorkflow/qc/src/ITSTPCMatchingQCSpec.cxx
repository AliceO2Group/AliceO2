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
#include "GlobalTracking/ITSTPCMatchingQCParams.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "CommonUtils/NameConf.h"
#include <TFile.h>

using namespace o2::framework;

namespace o2
{
namespace globaltracking
{

void ITSTPCMatchingQCDevice::init(InitContext& ic)
{
  const o2::globaltracking::ITSTPCMatchingQCParams* params = &o2::globaltracking::ITSTPCMatchingQCParams::Instance();

  mMatchITSTPCQC = std::make_unique<o2::globaltracking::MatchITSTPCQC>();
  mMatchITSTPCQC->init();
  mMatchITSTPCQC->setDataRequest(mDataRequest);
  mMatchITSTPCQC->setPtCut(params->minPtCut);
  mMatchITSTPCQC->setEtaCut(params->etaCut);
  mMatchITSTPCQC->setMinNTPCClustersCut(params->minNTPCClustersCut);
  mMatchITSTPCQC->setMinDCAtoBeamPipeDistanceCut(params->minDCACut);
  mMatchITSTPCQC->setMinDCAtoBeamPipeYCut(params->minDCACutY);
  if (mUseMC) {
    mMatchITSTPCQC->setUseMC(mUseMC);
  }
}

//_____________________________________________________________

void ITSTPCMatchingQCDevice::run(o2::framework::ProcessingContext& pc)
{

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
  output.snapshot(Output{"GLO", "ITSTPCMATCHQC", 0, Lifetime::Sporadic}, objar);

  TFile* f = new TFile(Form("outITSTPCmatchingQC.root"), "RECREATE");
  objar.Write("ObjArray", TObject::kSingleKey);
  f->Close();
}
} // namespace globaltracking

namespace framework
{
using GID = o2::dataformats::GlobalTrackID;

DataProcessorSpec getITSTPCMatchingQCDevice(bool useMC)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("GLO", "ITSTPCMATCHQC", 0, Lifetime::Sporadic);

  auto dataRequest = std::make_shared<o2::globaltracking::DataRequest>();
  GID::mask_t mSrc = GID::getSourcesMask("TPC,ITS-TPC");
  dataRequest->requestTracks(mSrc, useMC);
  return DataProcessorSpec{
    "itstpc-matching-qc",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<o2::globaltracking::ITSTPCMatchingQCDevice>(dataRequest, useMC)},
    Options{{}}};
}

} // namespace framework
} // namespace o2
