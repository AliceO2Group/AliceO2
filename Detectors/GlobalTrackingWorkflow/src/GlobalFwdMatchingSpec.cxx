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

/// @file   GlobalFwdMatchingSpec.cxx

#include <vector>
#include <string>
#include "TStopwatch.h"
#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "CommonUtils/StringUtils.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "DataFormatsMFT/TrackMFT.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ReconstructionDataFormats/GlobalFwdTrack.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsParameters/GRPObject.h"
#include "GlobalTracking/MatchGlobalFwd.h"
#include "GlobalTrackingWorkflow/GlobalFwdMatchingSpec.h"
#include "ITSMFTReconstruction/ClustererParam.h"

using namespace o2::framework;
using MCLabelsTr = gsl::span<const o2::MCCompLabel>;
using GTrackID = o2::dataformats::GlobalTrackID;

namespace o2
{
namespace globaltracking
{

class GlobalFwdMatchingDPL : public Task
{
 public:
  GlobalFwdMatchingDPL(std::shared_ptr<DataRequest> dr, bool useMC)
    : mDataRequest(dr), mUseMC(useMC) {}
  ~GlobalFwdMatchingDPL() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;

 private:
  std::shared_ptr<DataRequest> mDataRequest;
  o2::globaltracking::MatchGlobalFwd mMatching; // Forward matching engine
  o2::itsmft::TopologyDictionary mMFTDict;      // cluster patterns dictionary

  bool mUseMC = true;
  TStopwatch mTimer;
};

void GlobalFwdMatchingDPL::init(InitContext& ic)
{
  //-------- init geometry and field --------//
  o2::base::GeometryManager::loadGeometry();
  std::unique_ptr<o2::parameters::GRPObject> grp{o2::parameters::GRPObject::loadFrom()};
  mMatching.setMFTTriggered(!grp->isDetContinuousReadOut(o2::detectors::DetID::MFT));
  const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::MFT>::Instance();
  if (mMatching.isMFTTriggered()) {
    mMatching.setMFTROFrameLengthMUS(alpParams.roFrameLengthTrig / 1.e3); // MFT ROFrame duration in \mus
  } else {
    mMatching.setMFTROFrameLengthInBC(alpParams.roFrameLengthInBC); // MFT ROFrame duration in \mus
  }
  mMatching.setMCTruthOn(mUseMC);

  // set bunch filling. Eventually, this should come from CCDB
  const auto* digctx = o2::steer::DigitizationContext::loadFromFile();
  const auto& bcfill = digctx->getBunchFilling();
  mMatching.setBunchFilling(bcfill);

  std::string dictPath = o2::itsmft::ClustererParam<o2::detectors::DetID::MFT>::Instance().dictFilePath;
  std::string dictFile = o2::base::NameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::MFT, dictPath);
  if (o2::utils::Str::pathExists(dictFile)) {
    mMFTDict.readFromFile(dictFile);
    LOG(info) << "Forward track-matching is running with a provided MFT dictionary: " << dictFile;
  } else {
    LOG(info) << "Dictionary " << dictFile << " is absent, Matching expects MFT cluster patterns";
  }
  mMatching.setMFTDictionary(&mMFTDict);
  float matchPlaneZ = ic.options().get<float>("matchPlaneZ");
  mMatching.setMatchingPlaneZ(matchPlaneZ);
  std::string matchFcn = ic.options().get<std::string>("matchFcn");
  std::string cutFcn = ic.options().get<std::string>("cutFcn");

  mMatching.init(matchFcn, cutFcn);
}

void GlobalFwdMatchingDPL::run(ProcessingContext& pc)
{
  const auto* dh = o2::header::get<o2::header::DataHeader*>(pc.inputs().getFirstValid(true).header);
  LOG(info) << " startOrbit: " << dh->firstTForbit;
  mTimer.Start(false);

  RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());

  mMatching.run(recoData);

  pc.outputs().snapshot(Output{"GLO", "GLFWD", 0, Lifetime::Timeframe}, mMatching.getMatchedFwdTracks());
  if (mUseMC) {
    pc.outputs().snapshot(Output{"GLO", "GLFWD_MC", 0, Lifetime::Timeframe}, mMatching.getMatchLabels());
  }
  mTimer.Stop();
}

void GlobalFwdMatchingDPL::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "Forward matcher total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getGlobalFwdMatchingSpec(bool useMC)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();

  o2::dataformats::GlobalTrackID::mask_t src = o2::dataformats::GlobalTrackID::getSourcesMask("MFT,MCH");

  dataRequest->requestMFTClusters(false); // MFT clusters labels are not used
  dataRequest->requestTracks(src, useMC);

  outputs.emplace_back("GLO", "GLFWD", 0, Lifetime::Timeframe);

  if (useMC) {
    outputs.emplace_back("GLO", "GLFWD_MC", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "globalfwd-track-matcher",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<GlobalFwdMatchingDPL>(dataRequest, useMC)},
    Options{
      {"matchFcn", VariantType::String, "matchALL", {"Matching function (matchALL, ...)"}},
      {"cutFcn", VariantType::String, "cutDisabled", {"matching candicate cut"}},
      {"matchPlaneZ", o2::framework::VariantType::Float, -77.5f, {"Matching plane z position [-77.5]"}}}};
}

} // namespace globaltracking
} // namespace o2
