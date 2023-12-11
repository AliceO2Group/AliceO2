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

#include "TOFWorkflow/RecoWorkflowSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Framework/Task.h"
#include "Framework/SerializationMethods.h"
#include "Headers/DataHeader.h"
#include "DataFormatsTOF/Cluster.h"
#include "GlobalTracking/MatchTOF.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "CommonUtils/NameConf.h"
#include <gsl/span>
#include "TStopwatch.h"
#include "TPCCalibration/VDriftHelper.h"

// from FIT
#include "DataFormatsFT0/RecPoints.h"

#include <memory> // for make_shared, make_unique, unique_ptr
#include <vector>

using namespace o2::framework;

namespace o2
{
namespace tof
{

// use the tasking system of DPL
// just need to implement 2 special methods init + run (there is no need to inherit from anything)
class TOFDPLRecoWorkflowTask
{
  using evIdx = o2::dataformats::EvIndex<int, int>;
  using MatchOutputType = std::vector<o2::dataformats::MatchInfoTOF>;

  bool mUseMC = true;
  bool mUseFIT = false;

 public:
  explicit TOFDPLRecoWorkflowTask(std::shared_ptr<o2::base::GRPGeomRequest> gr, bool useMC, bool useFIT) : mGGCCDBRequest(gr), mUseMC(useMC), mUseFIT(useFIT) {}

  void init(framework::InitContext& ic)
  {
    o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
    mTimer.Stop();
    mTimer.Reset();
  }

  void run(framework::ProcessingContext& pc)
  {
    mTimer.Start(false);
    updateTimeDependentParams(pc);
    //>>>---------- attach input data --------------->>>
    const auto clustersRO = pc.inputs().get<gsl::span<o2::tof::Cluster>>("tofcluster");
    const auto tracksRO = pc.inputs().get<gsl::span<o2::dataformats::TrackTPCITS>>("globaltrack");

    if (mUseFIT) {
      // Note: the particular variable will go out of scope, but the span is passed by copy to the
      // worker and the underlying memory is valid throughout the whole computation
      auto recPoints = std::move(pc.inputs().get<gsl::span<o2::ft0::RecPoints>>("fitrecpoints"));
      mMatcher.setFITRecPoints(recPoints);
      LOG(info) << "TOF Reco Workflow pulled " << recPoints.size() << " FIT RecPoints";
    }

    // we do a copy of the input but we are looking for a way to avoid it (current problem in conversion form unique_ptr to *)

    gsl::span<const o2::MCCompLabel> itstpclab;
    o2::dataformats::MCTruthContainer<o2::MCCompLabel> toflab;
    if (mUseMC) {
      const auto toflabel = pc.inputs().get<o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("tofclusterlabel");
      itstpclab = pc.inputs().get<gsl::span<o2::MCCompLabel>>("itstpclabel");
      toflab = std::move(*toflabel);
    }

    mMatcher.run(tracksRO, clustersRO, toflab, itstpclab);

    // in run_match_tof aggiugnere esplicitamente la chiamata a fill del tree (nella classe MatchTOF) e il metodo per leggere i vettori di output

    //...
    // LOG(info) << "TOF CLUSTERER : TRANSFORMED " << digits->size()
    //           << " DIGITS TO " << mClustersArray.size() << " CLUSTERS";

    // send matching-info
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MTC_ITSTPC", 0}, mMatcher.getMatchedTrackVector());
    if (mUseMC) {
      pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "MCMATCHTOF", 0}, mMatcher.getMatchedTOFLabelsVector());
    }
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "CALIBDATA", 0}, mMatcher.getCalibVector());
    mTimer.Stop();
  }

  void endOfStream(EndOfStreamContext& ec)
  {
    LOGF(info, "TOF Matching total timing: Cpu: %.3e Real: %.3e s in %d slots",
         mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
  }

  void updateTimeDependentParams(ProcessingContext& pc)
  {
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    mTPCVDriftHelper.extractCCDBInputs(pc);
    static bool initOnceDone = false;
    if (!initOnceDone) { // this params need to be queried only once
      initOnceDone = true;
      // put here init-once stuff
    }
    // we may have other params which need to be queried regularly
    if (mTPCVDriftHelper.isUpdated()) {
      LOGP(info, "Updating TPC fast transform map with new VDrift factor of {} wrt reference {} from source {}",
           mTPCVDriftHelper.getVDriftObject().corrFact, mTPCVDriftHelper.getVDriftObject().refVDrift, mTPCVDriftHelper.getSourceName());
      mMatcher.setTPCVDrift(mTPCVDriftHelper.getVDriftObject());
      mTPCVDriftHelper.acknowledgeUpdate();
    }
  }

  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
  {
    if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
      return;
    }
    if (mTPCVDriftHelper.accountCCDBInputs(matcher, obj)) {
      return;
    }
  }

 private:
  o2::globaltracking::MatchTOF mMatcher; ///< Cluster finder
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  o2::tpc::VDriftHelper mTPCVDriftHelper{};
  TStopwatch mTimer;
};

o2::framework::DataProcessorSpec getTOFRecoWorkflowSpec(bool useMC, bool useFIT)
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;
  inputs.emplace_back("tofcluster", o2::header::gDataOriginTOF, "CLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("globaltrack", "GLO", "TPCITS", 0, Lifetime::Timeframe);
  if (useMC) {
    inputs.emplace_back("tofclusterlabel", o2::header::gDataOriginTOF, "CLUSTERSMCTR", 0, Lifetime::Timeframe);
    inputs.emplace_back("itstpclabel", "GLO", "TPCITS_MC", 0, Lifetime::Timeframe);
  }

  if (useFIT) {
    inputs.emplace_back("fitrecpoints", o2::header::gDataOriginFT0, "RECPOINTS", 0, Lifetime::Timeframe);
  }
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              inputs,
                                                              true);
  o2::tpc::VDriftHelper::requestCCDBInputs(inputs);

  outputs.emplace_back(o2::header::gDataOriginTOF, "MTC_ITSTPC", 0, Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back(o2::header::gDataOriginTOF, "MCMATCHTOF", 0, Lifetime::Timeframe);
  }
  outputs.emplace_back(o2::header::gDataOriginTOF, "CALIBDATA", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "TOFRecoWorkflow",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TOFDPLRecoWorkflowTask>(ggRequest, useMC, useFIT)},
    Options{
      {"material-lut-path", VariantType::String, "", {"Path of the material LUT file"}}}};
}

} // end namespace tof
} // end namespace o2
