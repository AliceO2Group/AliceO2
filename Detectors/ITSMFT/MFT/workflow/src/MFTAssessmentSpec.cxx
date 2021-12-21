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

/// @file MFTAssessmentSpec.cxx

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "MFTWorkflow/MFTAssessmentSpec.h"
#include "CommonUtils/NameConf.h"
#include <TFile.h>

using namespace o2::framework;

namespace o2
{
namespace mft
{

//_____________________________________________________________
void MFTAssessmentSpec::init(InitContext& ic)
{
  mMFTAssessment = std::make_unique<o2::mft::MFTAssessment>();
  mMFTAssessment->init();
  if (mUseMC) {
    mMFTAssessment->setUseMC(mUseMC);
  }
  mMFTAssessment->setGRPFileName(o2::base::NameConf::getGRPFileName());
  mMFTAssessment->setGeomFileName(o2::base::NameConf::getGeomFileName());
}

//_____________________________________________________________
void MFTAssessmentSpec::run(o2::framework::ProcessingContext& pc)
{

  mMFTAssessment->run(pc);
}

//_____________________________________________________________
void MFTAssessmentSpec::endOfStream(o2::framework::EndOfStreamContext& ec)
{

  mMFTAssessment->finalize();
  sendOutput(ec.outputs());
}

//_____________________________________________________________
void MFTAssessmentSpec::sendOutput(DataAllocator& output)
{

  TObjArray objar;
  mMFTAssessment->getHistos(objar);

  output.snapshot(Output{"MFT", "MFTASSESSMENT", 0, Lifetime::Timeframe}, objar);

  TFile* f = new TFile(Form("MFTAssessment.root"), "RECREATE");
  objar.Write("ObjArray", TObject::kSingleKey);
  f->Close();
}
} // namespace mft

namespace framework
{

//_____________________________________________________________
DataProcessorSpec getMFTAssessmentSpec(bool useMC)
{

  std::vector<InputSpec> inputs;
  inputs.emplace_back("compClusters", "MFT", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("patterns", "MFT", "PATTERNS", 0, Lifetime::Timeframe);
  inputs.emplace_back("clustersrofs", "MFT", "CLUSTERSROF", 0, Lifetime::Timeframe);
  inputs.emplace_back("tracksrofs", "MFT", "MFTTrackROF", 0, Lifetime::Timeframe);
  inputs.emplace_back("tracks", "MFT", "TRACKS", 0, Lifetime::Timeframe);

  if (useMC) {
    inputs.emplace_back("clslabels", "MFT", "CLUSTERSMCTR", 0, Lifetime::Timeframe);
    inputs.emplace_back("MCCls2ROF", "MFT", "CLUSTERSMC2ROF", 0, Lifetime::Timeframe);
    inputs.emplace_back("trklabels", "MFT", "TRACKSMCTR", 0, Lifetime::Timeframe);
    inputs.emplace_back("MCTrk2ROF", "MFT", "TRACKSMC2ROF", 0, Lifetime::Timeframe);
  }

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("MFT", "MFTASSESSMENT", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "mft-assessment",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<o2::mft::MFTAssessmentSpec>(useMC)},
    Options{{}}};
}

} // namespace framework
} // namespace o2
