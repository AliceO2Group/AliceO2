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

/// @file GlobalFwdAssessmentSpec.cxx

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "GlobalTrackingWorkflow/GlobalFwdMatchingAssessmentSpec.h"
#include "DetectorsBase/Propagator.h"
#include "Field/MagneticField.h"
#include "TGeoGlobalMagField.h"
#include "CommonUtils/NameConf.h"
#include <TFile.h>

using namespace o2::framework;

namespace o2
{
namespace globaltracking
{

//_____________________________________________________________
void GlobalFwdAssessmentSpec::init(InitContext& ic)
{
  mGloFwdAssessment = std::make_unique<o2::globaltracking::GloFwdAssessment>(mUseMC);

  if (mMIDFilterDisabled) {
    mGloFwdAssessment->disableMIDFilter();
  }
  auto filename = o2::base::NameConf::getGRPFileName();
  const auto grp = o2::parameters::GRPObject::loadFrom(filename.c_str());
  if (grp) {
    o2::base::Propagator::initFieldFromGRP(grp);
    auto field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
    double centerMFT[3] = {0, 0, -61.4}; // Field at center of MFT
    auto Bz = field->getBz(centerMFT);
    mGloFwdAssessment->setBz(Bz);
  }

  for (int sw = 0; sw < NStopWatches; sw++) {
    mTimer[sw].Stop();
    mTimer[sw].Reset();
  }

  mTimer[SWTot].Start(false);
  mGloFwdAssessment->init(mFinalizeAnalysis);
}

//_____________________________________________________________
void GlobalFwdAssessmentSpec::run(o2::framework::ProcessingContext& pc)
{

  mTimer[SWQCAsync].Start(false);
  mGloFwdAssessment->runBasicQC(pc);
  mTimer[SWQCAsync].Stop();

  if (mUseMC) {

    mTimer[SWTrackables].Start(false);
    mGloFwdAssessment->processPairables();
    mTimer[SWTrackables].Stop();

    if (mProcessGen) {
      mTimer[SWGenerated].Start(false);
      mGloFwdAssessment->processGeneratedTracks();
      mTimer[SWGenerated].Stop();
    }
    mTimer[SWRecoAndTrue].Start(false);
    mGloFwdAssessment->processRecoAndTrueTracks();
    mTimer[SWRecoAndTrue].Stop();
  }
}

//_____________________________________________________________
void GlobalFwdAssessmentSpec::endOfStream(o2::framework::EndOfStreamContext& ec)
{
  if (mFinalizeAnalysis) {
    mTimer[SWAnalysis].Start(false);
    mGloFwdAssessment->finalizeAnalysis();
    mTimer[SWAnalysis].Stop();
  }

  sendOutput(ec.outputs());
  mTimer[SWTot].Stop();

  for (int i = 0; i < NStopWatches; i++) {
    LOGF(info, "Timing %18s: Cpu: %.3e s; Real: %.3e s in %d slots", TimerName[i], mTimer[i].CpuTime(), mTimer[i].RealTime(), mTimer[i].Counter() - 1);
  }
}

//_____________________________________________________________
void GlobalFwdAssessmentSpec::sendOutput(DataAllocator& output)
{
  TObjArray objar;
  mGloFwdAssessment->getHistos(objar);

  output.snapshot(Output{"GLO", "FWDASSESSMENT", 0, Lifetime::Sporadic}, objar);

  TFile* f = new TFile(Form("GlobalForwardAssessment.root"), "RECREATE");
  objar.Write();
  f->Close();
}

//_____________________________________________________________
DataProcessorSpec getGlobaFwdAssessmentSpec(bool useMC, bool processGen, bool midFilterDisabled, bool finalizeAnalysis)
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;

  inputs.emplace_back("fwdtracks", "GLO", "GLFWD", 0, Lifetime::Timeframe);
  inputs.emplace_back("mfttracks", "MFT", "TRACKS", 0, Lifetime::Timeframe);
  inputs.emplace_back("mchtracks", "MCH", "TRACKS", 0, Lifetime::Timeframe);

  if (useMC) {
    inputs.emplace_back("mfttrklabels", "MFT", "TRACKSMCTR", 0, Lifetime::Timeframe);
    inputs.emplace_back("mchtrklabels", "MCH", "TRACKLABELS", 0, Lifetime::Timeframe);
    inputs.emplace_back("fwdtrklabels", "GLO", "GLFWD_MC", 0, Lifetime::Timeframe);
  }

  outputs.emplace_back("GLO", "FWDASSESSMENT", 0, Lifetime::Sporadic);

  return DataProcessorSpec{
    "glofwd-assessment",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<o2::globaltracking::GlobalFwdAssessmentSpec>(useMC, processGen, midFilterDisabled, finalizeAnalysis)},
    Options{{}}};
}

} // namespace globaltracking
} // namespace o2
