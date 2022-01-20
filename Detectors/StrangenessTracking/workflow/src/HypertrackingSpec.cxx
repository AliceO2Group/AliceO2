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

#include "TGeoGlobalMagField.h"
#include "DataFormatsParameters/GRPObject.h"
#include "Framework/ConfigParamRegistry.h"
#include "Field/MagneticField.h"

#include "StrangenessTrackingWorkflow/HypertrackingSpec.h"
#include "ITSWorkflow/ClusterWriterSpec.h"
#include "ITSWorkflow/TrackerSpec.h"
#include "ITSWorkflow/TrackReaderSpec.h"
#include "ITSMFTWorkflow/ClusterReaderSpec.h"
#include "GlobalTrackingWorkflowReaders/SecondaryVertexReaderSpec.h"
#include "GlobalTrackingWorkflowReaders/TrackTPCITSReaderSpec.h"
#include "GlobalTrackingWorkflow/TOFMatcherSpec.h"

#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITS/TrackITS.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "ITStracking/IOUtils.h"
#include "ITSMFTReconstruction/ClustererParam.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"

#include "StrangenessTracking/HyperTracker.h"

#include <fmt/format.h>

namespace o2
{
using namespace o2::framework;
namespace strangeness_tracking
{

class HypertrackerSpec : public framework::Task
{
 public:
  using ITSCluster = o2::BaseCluster<float>;

  HypertrackerSpec(bool isMC = false);
  ~HypertrackerSpec() override = default;

  void init(framework::InitContext& ic) final;
  void run(framework::ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;

 private:
  bool mIsMC = false;
  bool mRecreateV0 = true;
  TStopwatch mTimer;
  HyperTracker mTracker;
  std::unique_ptr<parameters::GRPObject> mGRP = nullptr;
};

framework::WorkflowSpec getWorkflow(bool useMC, bool useRootInput)
{
  framework::WorkflowSpec specs;
  if (useRootInput) {
    specs.emplace_back(o2::itsmft::getITSClusterReaderSpec(useMC, true));
    specs.emplace_back(o2::its::getITSTrackReaderSpec(useMC));
    specs.emplace_back(o2::vertexing::getSecondaryVertexReaderSpec());
    specs.emplace_back(o2::globaltracking::getTrackTPCITSReaderSpec(true));
    // auto src = o2::dataformats::GlobalTrackID::Source::ITSTPCTOF | o2::dataformats::GlobalTrackID::Source::ITSTPC | o2::dataformats::GlobalTrackID::Source::TPCTOF;
    // specs.emplace_back(o2::globaltracking::getTOFMatcherSpec(src, true, false, false, 0));
  }
  specs.emplace_back(getHyperTrackerSpec());
  return specs;
}

HypertrackerSpec::HypertrackerSpec(bool isMC) : mIsMC{isMC}
{
  // no ops
}

void HypertrackerSpec::init(framework::InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();

  auto filename = ic.options().get<std::string>("grp-file");
  const auto grp = parameters::GRPObject::loadFrom(filename);

  // load propagator
  base::Propagator::initFieldFromGRP(grp);
  std::string matLUTPath = ic.options().get<std::string>("material-lut-path");
  std::string matLUTFile = o2::base::NameConf::getMatLUTFileName(matLUTPath);
  if (o2::utils::Str::pathExists(matLUTFile)) {
    auto* lut = o2::base::MatLayerCylSet::loadFromFile(matLUTFile);
    o2::base::Propagator::Instance()->setMatLUT(lut);
    mTracker.setCorrType(o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrLUT);
    LOG(info) << "Loaded material LUT from " << matLUTFile;
  } else {
    LOG(info) << "Material LUT " << matLUTFile << " file is absent, only heuristic material correction can be used";
  }

  // load geometry
  base::GeometryManager::loadGeometry();
  auto gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));

  LOG(info) << "Initialized Hypertracker...";
}

void HypertrackerSpec::run(framework::ProcessingContext& pc)
{
  mTimer.Start(false);
  LOG(info) << "Running Hypertracker...";
  // ITS
  auto ITSclus = pc.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>("compClusters");
  auto ITSpatt = pc.inputs().get<gsl::span<unsigned char>>("patterns");
  auto ITStracks = pc.inputs().get<gsl::span<o2::its::TrackITS>>("ITSTrack");
  auto ROFsInput = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("ROframes");
  auto ITSTrackClusIdx = pc.inputs().get<gsl::span<int>>("trackITSClIdx");

  // V0
  auto v0vec = pc.inputs().get<gsl::span<o2::dataformats::V0>>("v0s");
  auto tpcITSTracks = pc.inputs().get<gsl::span<o2::dataformats::TrackTPCITS>>("trackTPCITS");

  // Monte Carlo
  auto labITSTPC = pc.inputs().get<gsl::span<o2::MCCompLabel>>("trackITSTPCMCTR");
  // auto labTPCTOF = pc.inputs().get<gsl::span<o2::MCCompLabel>>("clsTOF_TPC_MCTR");
  // auto labITSTPCTOF = pc.inputs().get<gsl::span<o2::MCCompLabel>>("clsTOF_GLO_MCTR");
  auto labITS = pc.inputs().get<gsl::span<o2::MCCompLabel>>("trackITSMCTR");
  LOGF(info, "ITSclus: %d \nITSpatt: %d \nITStracks: %d \nROFsInput: %d \nITSTrackClusIdx: %d \nTPCITStracks: %d \nv0s: %d \nlabITSTPC: %d\nlabITS: %d",
       ITSclus.size(),
       ITSpatt.size(),
       ITStracks.size(),
       ROFsInput.size(),
       ITSTrackClusIdx.size(),
       tpcITSTracks.size(),
       v0vec.size(),
       labITSTPC.size(),
       //  labTPCTOF.size(),
       //  labITSTPCTOF.size(),
       labITS.size());
  //  \nlabTPCTOF: %d\nlabITSTPCTOF: %d

  // ITS dict
  o2::itsmft::TopologyDictionary ITSdict;
  std::string dictPath = o2::itsmft::ClustererParam<o2::detectors::DetID::ITS>::Instance().dictFilePath;
  std::string dictFile = o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, dictPath);
  ITSdict.readFromFile(dictFile);

  auto pattIt = ITSpatt.begin();
  std::vector<ITSCluster> ITSClustersArray;
  ITSClustersArray.reserve(ITSclus.size());
  o2::its::ioutils::convertCompactClusters(ITSclus, pattIt, ITSClustersArray, ITSdict);

  auto geom = o2::its::GeometryTGeo::Instance();
  auto field = static_cast<field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
  double origD[3] = {0., 0., 0.};
  mTracker.setBz(field->getBz(origD));

  mTracker.loadData(ITStracks, ITSClustersArray, ITSTrackClusIdx, v0vec, geom);
  mTracker.process();

  pc.outputs().snapshot(Output{"HYP", "V0S", 0, Lifetime::Timeframe}, mTracker.getV0());
  pc.outputs().snapshot(Output{"HYP", "HYPERTRACKS", 0, Lifetime::Timeframe}, mTracker.getHyperTracks());
  pc.outputs().snapshot(Output{"HYP", "CHI2", 0, Lifetime::Timeframe}, mTracker.getChi2vec());
  pc.outputs().snapshot(Output{"HYP", "ITSREFS", 0, Lifetime::Timeframe}, mTracker.getITStrackRef());

  mTimer.Stop();
}

void HypertrackerSpec::endOfStream(framework::EndOfStreamContext& ec)
{
  LOGF(info, "Hypertracker total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getHyperTrackerSpec()
{
  std::vector<InputSpec> inputs;

  // ITS
  inputs.emplace_back("compClusters", "ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("patterns", "ITS", "PATTERNS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", "ITS", "CLUSTERSROF", 0, Lifetime::Timeframe);
  inputs.emplace_back("trackITSClIdx", "ITS", "TRACKCLSID", 0, Lifetime::Timeframe);
  inputs.emplace_back("ITSTrack", "ITS", "TRACKS", 0, Lifetime::Timeframe);

  // V0
  inputs.emplace_back("v0s", "GLO", "V0S", 0, Lifetime::Timeframe);                // found V0s
  inputs.emplace_back("v02pvrf", "GLO", "PVTX_V0REFS", 0, Lifetime::Timeframe);    // prim.vertex -> V0s refs
  inputs.emplace_back("cascs", "GLO", "CASCS", 0, Lifetime::Timeframe);            // found Cascades
  inputs.emplace_back("cas2pvrf", "GLO", "PVTX_CASCREFS", 0, Lifetime::Timeframe); // prim.vertex -> Cascades refs

  // TPC-ITS
  inputs.emplace_back("trackTPCITS", "GLO", "TPCITS", 0, Lifetime::Timeframe);

  // TPC-TOF
  // inputs.emplace_back("matchTPCTOF", "TOF", "MTC_TPC", 0, Lifetime::Timeframe); // Matching input type manually set to 0

  // Monte Carlo
  inputs.emplace_back("trackITSTPCMCTR", "GLO", "TPCITS_MC", 0, Lifetime::Timeframe); // MC truth
  // inputs.emplace_back("clsTOF_GLO_MCTR", "TOF", "MCMTC_ITSTPC", 0, Lifetime::Timeframe); // MC truth
  inputs.emplace_back("trackITSMCTR", "ITS", "TRACKSMCTR", 0, Lifetime::Timeframe); // MC truth
  // inputs.emplace_back("clsTOF_TPC_MCTR", "TOF", "MCMTC_TPC", 0, Lifetime::Timeframe);    // MC truth, // Matching input type manually set to 0

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("HYP", "V0S", 0, Lifetime::Timeframe);
  outputs.emplace_back("HYP", "HYPERTRACKS", 0, Lifetime::Timeframe);

  outputs.emplace_back("HYP", "CHI2", 0, Lifetime::Timeframe);
  outputs.emplace_back("HYP", "ITSREFS", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "hypertracker",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<HypertrackerSpec>()},
    Options{
      {"grp-file", VariantType::String, "o2sim_grp.root", {"Name of the grp file"}},
      {"material-lut-path", VariantType::String, "", {"Path of the material LUT file"}}}};
}

} // namespace strangeness_tracking
} // namespace o2