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
void GlobalFwdAssessmentSpec::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    auto field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
    const double centerMFT[3] = {0, 0, -61.4}; // Field at center of MFT
    auto Bz = field->getBz(centerMFT);
    mGloFwdAssessment->setBz(Bz);
    return;
  }
  return;
}

//_____________________________________________________________
void GlobalFwdAssessmentSpec::init(InitContext& ic)
{
  mGloFwdAssessment = std::make_unique<o2::globaltracking::GloFwdAssessment>(mUseMC);
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  if (mMIDFilterDisabled) {
    mGloFwdAssessment->disableMIDFilter();
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
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);

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
    mGloFwdAssessment->processRecoTracks();
    mGloFwdAssessment->processTrueTracks();
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

  output.snapshot(Output{"GLO", "FWDASSESSMENT", 0}, objar);

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

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                          // orbitResetTime
                                                              false,                          // GRPECS=true
                                                              false,                          // GRPLHCIF
                                                              true,                           // GRPMagField
                                                              false,                          // askMatLUT
                                                              o2::base::GRPGeomRequest::None, // geometry
                                                              inputs,
                                                              true); // query only once all objects except mag.field

  outputs.emplace_back("GLO", "FWDASSESSMENT", 0, Lifetime::Sporadic);

  return DataProcessorSpec{
    "glofwd-assessment",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<o2::globaltracking::GlobalFwdAssessmentSpec>(useMC, processGen, ggRequest, midFilterDisabled, finalizeAnalysis)},
    Options{{}}};
}

} // namespace globaltracking
} // namespace o2
