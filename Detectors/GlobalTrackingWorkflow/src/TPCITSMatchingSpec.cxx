// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TPCITSMatchingSpec.cxx

#include <vector>

#include "GlobalTracking/MatchTPCITS.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsTPC/Constants.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include <string>
#include "TStopwatch.h"
#include "Framework/ConfigParamRegistry.h"
#include "GlobalTrackingWorkflow/TPCITSMatchingSpec.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "GlobalTracking/MatchTPCITSParams.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DataFormatsParameters/GRPObject.h"
#include "Headers/DataHeader.h"
#include "CommonDataFormat/BunchFilling.h"
#include "CommonDataFormat/FlatHisto2D.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"

using namespace o2::framework;
using MCLabelsCl = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
using MCLabelsTr = gsl::span<const o2::MCCompLabel>;
using GTrackID = o2::dataformats::GlobalTrackID;

namespace o2
{
namespace globaltracking
{

class TPCITSMatchingDPL : public Task
{
 public:
  TPCITSMatchingDPL(std::shared_ptr<DataRequest> dr, bool useFT0, bool calib, bool skipTPCOnly, bool useMC)
    : mDataRequest(dr), mUseFT0(useFT0), mCalibMode(calib), mSkipTPCOnly(skipTPCOnly), mUseMC(useMC) {}
  ~TPCITSMatchingDPL() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;

 private:
  std::shared_ptr<DataRequest> mDataRequest;
  o2::globaltracking::MatchTPCITS mMatching; // matching engine
  o2::itsmft::TopologyDictionary mITSDict;   // cluster patterns dictionary
  bool mUseFT0 = false;
  bool mCalibMode = false;
  bool mSkipTPCOnly = false; // to use only externally constrained tracks (for test only)
  bool mUseMC = true;
  TStopwatch mTimer;
};

void TPCITSMatchingDPL::init(InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();
  //-------- init geometry and field --------//
  o2::base::GeometryManager::loadGeometry();
  o2::base::Propagator::initFieldFromGRP();
  std::unique_ptr<o2::parameters::GRPObject> grp{o2::parameters::GRPObject::loadFrom()};
  mMatching.setSkipTPCOnly(mSkipTPCOnly);
  mMatching.setITSTriggered(!grp->isDetContinuousReadOut(o2::detectors::DetID::ITS));
  const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
  if (mMatching.isITSTriggered()) {
    mMatching.setITSROFrameLengthMUS(alpParams.roFrameLengthTrig / 1.e3); // ITS ROFrame duration in \mus
  } else {
    mMatching.setITSROFrameLengthInBC(alpParams.roFrameLengthInBC); // ITS ROFrame duration in \mus
  }
  mMatching.setMCTruthOn(mUseMC);
  mMatching.setUseFT0(mUseFT0);
  mMatching.setVDriftCalib(mCalibMode);
  //
  std::string dictPath = ic.options().get<std::string>("its-dictionary-path");
  std::string dictFile = o2::base::NameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, dictPath, ".bin");
  if (o2::utils::Str::pathExists(dictFile)) {
    mITSDict.readBinaryFile(dictFile);
    LOG(INFO) << "Matching is running with a provided ITS dictionary: " << dictFile;
  } else {
    LOG(INFO) << "Dictionary " << dictFile << " is absent, Matching expects ITS cluster patterns";
  }
  mMatching.setITSDictionary(&mITSDict);

  // this is a hack to provide Mat.LUT from the local file, in general will be provided by the framework from CCDB
  std::string matLUTPath = ic.options().get<std::string>("material-lut-path");
  std::string matLUTFile = o2::base::NameConf::getMatLUTFileName(matLUTPath);
  if (o2::utils::Str::pathExists(matLUTFile)) {
    auto* lut = o2::base::MatLayerCylSet::loadFromFile(matLUTFile);
    o2::base::Propagator::Instance()->setMatLUT(lut);
    LOG(INFO) << "Loaded material LUT from " << matLUTFile;
  } else {
    LOG(INFO) << "Material LUT " << matLUTFile << " file is absent, only TGeo can be used";
  }

  int dbgFlags = ic.options().get<int>("debug-tree-flags");
  mMatching.setDebugFlag(dbgFlags);

  // set bunch filling. Eventually, this should come from CCDB
  const auto* digctx = o2::steer::DigitizationContext::loadFromFile("collisioncontext.root");
  const auto& bcfill = digctx->getBunchFilling();
  mMatching.setBunchFilling(bcfill);

  mMatching.init();
  //
}

void TPCITSMatchingDPL::run(ProcessingContext& pc)
{
  const auto* dh = o2::header::get<o2::header::DataHeader*>(pc.inputs().getByPos(0).header);
  LOG(INFO) << " startOrbit: " << dh->firstTForbit;
  mTimer.Start(false);
  RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());

  mMatching.run(recoData);

  pc.outputs().snapshot(Output{"GLO", "TPCITS", 0, Lifetime::Timeframe}, mMatching.getMatchedTracks());
  if (mUseMC) {
    pc.outputs().snapshot(Output{"GLO", "TPCITS_MC", 0, Lifetime::Timeframe}, mMatching.getMatchLabels());
  }

  if (mCalibMode) {
    auto* hdtgl = mMatching.getHistoDTgl();
    pc.outputs().snapshot(Output{"GLO", "TPCITS_VDHDTGL", 0, Lifetime::Timeframe}, (*hdtgl).getBase());
    hdtgl->clear();
  }
  mTimer.Stop();
}

void TPCITSMatchingDPL::endOfStream(EndOfStreamContext& ec)
{
  mMatching.end();
  LOGF(INFO, "TPC-ITS matching total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getTPCITSMatchingSpec(GTrackID::mask_t src, bool useFT0, bool calib, bool skipTPCOnly, bool useMC)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();

  dataRequest->requestTracks(src, useMC);
  dataRequest->requestClusters(src, false); // no MC labels for clusters needed for refit only

  if (useFT0) {
    dataRequest->requestFT0RecPoints(false);
  }
  outputs.emplace_back("GLO", "TPCITS", 0, Lifetime::Timeframe);

  if (calib) {
    outputs.emplace_back("GLO", "TPCITS_VDHDTGL", 0, Lifetime::Timeframe);
  }

  if (useMC) {
    dataRequest->inputs.emplace_back("clusITSMCTR", "ITS", "CLUSTERSMCTR", 0, Lifetime::Timeframe); // for afterburner
    outputs.emplace_back("GLO", "TPCITS_MC", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "itstpc-track-matcher",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCITSMatchingDPL>(dataRequest, useFT0, calib, skipTPCOnly, useMC)},
    Options{
      {"its-dictionary-path", VariantType::String, "", {"Path of the cluster-topology dictionary file"}},
      {"material-lut-path", VariantType::String, "", {"Path of the material LUT file"}},
      {"debug-tree-flags", VariantType::Int, 0, {"DebugFlagTypes bit-pattern for debug tree"}}}};
}

} // namespace globaltracking
} // namespace o2
