// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CosmicsMatchingSpec.cxx

#include <vector>
#include <string>
#include "TStopwatch.h"
#include "GlobalTracking/MatchCosmics.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsTPC/Constants.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "Framework/ConfigParamRegistry.h"
#include "GlobalTrackingWorkflow/CosmicsMatchingSpec.h"
#include "ReconstructionDataFormats/GlobalTrackAccessor.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/MatchInfoTOF.h"
#include "ReconstructionDataFormats/TrackTPCTOF.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DataFormatsParameters/GRPObject.h"
#include "Headers/DataHeader.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "Framework/Task.h"

using namespace o2::framework;
using MCLabelsTr = gsl::span<const o2::MCCompLabel>;
using GTrackID = o2::dataformats::GlobalTrackID;
using DetID = o2::detectors::DetID;

namespace o2
{
namespace globaltracking
{

class CosmicsMatchingSpec : public Task
{
 public:
  CosmicsMatchingSpec(std::shared_ptr<DataRequest> dr, bool useMC) : mDataRequest(dr), mUseMC(useMC) {}
  ~CosmicsMatchingSpec() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;

 private:
  std::shared_ptr<DataRequest> mDataRequest;
  o2::globaltracking::MatchCosmics mMatching; // matching engine
  bool mUseMC = true;
  TStopwatch mTimer;
};

void CosmicsMatchingSpec::init(InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();
  //-------- init geometry and field --------//
  o2::base::GeometryManager::loadGeometry();
  o2::base::Propagator::initFieldFromGRP();
  std::unique_ptr<o2::parameters::GRPObject> grp{o2::parameters::GRPObject::loadFrom()};
  const auto& alpParams = o2::itsmft::DPLAlpideParam<DetID::ITS>::Instance();
  if (!grp->isDetContinuousReadOut(DetID::ITS)) {
    mMatching.setITSROFrameLengthMUS(alpParams.roFrameLengthTrig / 1.e3); // ITS ROFrame duration in \mus
  } else {
    mMatching.setITSROFrameLengthMUS(alpParams.roFrameLengthInBC * o2::constants::lhc::LHCBunchSpacingNS * 1e-3); // ITS ROFrame duration in \mus
  }
  //
  std::string dictPath = ic.options().get<std::string>("its-dictionary-path");
  std::string dictFile = o2::base::NameConf::getAlpideClusterDictionaryFileName(DetID::ITS, dictPath, ".bin");
  auto itsDict = std::make_unique<o2::itsmft::TopologyDictionary>();
  if (o2::utils::Str::pathExists(dictFile)) {
    itsDict->readBinaryFile(dictFile);
    LOG(INFO) << "Matching is running with a provided ITS dictionary: " << dictFile;
  } else {
    LOG(INFO) << "Dictionary " << dictFile << " is absent, Matching expects ITS cluster patterns";
  }
  o2::its::GeometryTGeo::Instance()->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2GRot) | o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L));
  mMatching.setITSDict(itsDict);

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

  mMatching.setDebugFlag(ic.options().get<int>("debug-tree-flags"));

  mMatching.setUseMC(mUseMC);
  mMatching.init();
  //
}

void CosmicsMatchingSpec::run(ProcessingContext& pc)
{
  mTimer.Start(false);

  RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());

  mMatching.process(recoData);
  pc.outputs().snapshot(Output{"GLO", "COSMICTRC", 0, Lifetime::Timeframe}, mMatching.getCosmicTracks());
  if (mUseMC) {
    pc.outputs().snapshot(Output{"GLO", "COSMICTRC_MC", 0, Lifetime::Timeframe}, mMatching.getCosmicTracksLbl());
  }
  mTimer.Stop();
}

void CosmicsMatchingSpec::endOfStream(EndOfStreamContext& ec)
{
  mMatching.end();
  LOGF(INFO, "Cosmics matching total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getCosmicsMatchingSpec(GTrackID::mask_t src, bool useMC)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();

  dataRequest->requestTracks(src, useMC);
  dataRequest->requestClusters(src, false); // no MC labels for clusters needed for refit only

  outputs.emplace_back("GLO", "COSMICTRC", 0, Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back("GLO", "COSMICTRC_MC", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "cosmics-matcher",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<CosmicsMatchingSpec>(dataRequest, useMC)},
    Options{
      {"its-dictionary-path", VariantType::String, "", {"Path of the cluster-topology dictionary file"}},
      {"material-lut-path", VariantType::String, "", {"Path of the material LUT file"}},
      {"debug-tree-flags", VariantType::Int, 0, {"DebugFlagTypes bit-pattern for debug tree"}}}};
}

} // namespace globaltracking
} // namespace o2
