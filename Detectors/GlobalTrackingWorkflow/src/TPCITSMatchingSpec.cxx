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
#include "ITStracking/IOUtils.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DataFormatsParameters/GRPObject.h"
#include "Headers/DataHeader.h"
#include "CommonDataFormat/BunchFilling.h"
#include "CommonDataFormat/FlatHisto2D.h"

// RSTODO to remove once the framework will start propagating the header.firstTForbit
#include "DetectorsRaw/HBFUtils.h"

using namespace o2::framework;
using MCLabelsCl = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
using MCLabelsTr = gsl::span<const o2::MCCompLabel>;

namespace o2
{
namespace globaltracking
{

void TPCITSMatchingDPL::init(InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();
  //-------- init geometry and field --------//
  o2::base::GeometryManager::loadGeometry();
  o2::base::Propagator::initFieldFromGRP(o2::base::NameConf::getGRPFileName());
  std::unique_ptr<o2::parameters::GRPObject> grp{o2::parameters::GRPObject::loadFrom(o2::base::NameConf::getGRPFileName())};
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
  std::string dictFile = o2::base::NameConf::getDictionaryFileName(o2::detectors::DetID::ITS, dictPath, ".bin");
  if (o2::base::NameConf::pathExists(dictFile)) {
    mITSDict.readBinaryFile(dictFile);
    LOG(INFO) << "Matching is running with a provided ITS dictionary: " << dictFile;
  } else {
    LOG(INFO) << "Dictionary " << dictFile << " is absent, Matching expects ITS cluster patterns";
  }

  // this is a hack to provide Mat.LUT from the local file, in general will be provided by the framework from CCDB
  std::string matLUTPath = ic.options().get<std::string>("material-lut-path");
  std::string matLUTFile = o2::base::NameConf::getMatLUTFileName(matLUTPath);
  if (o2::base::NameConf::pathExists(matLUTFile)) {
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
  mTimer.Start(false);
  const auto tracksITS = pc.inputs().get<gsl::span<o2::its::TrackITS>>("trackITS");
  const auto trackClIdxITS = pc.inputs().get<gsl::span<int>>("trackClIdx");
  const auto tracksITSROF = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("trackITSROF");
  const auto clusITS = pc.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>("clusITS");
  const auto clusITSROF = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("clusITSROF");
  const auto patterns = pc.inputs().get<gsl::span<unsigned char>>("clusITSPatt");
  const auto tracksTPC = pc.inputs().get<gsl::span<o2::tpc::TrackTPC>>("trackTPC");
  const auto tracksTPCClRefs = pc.inputs().get<gsl::span<o2::tpc::TPCClRefElem>>("trackTPCClRefs");

  const auto clusTPCShmap = pc.inputs().get<gsl::span<unsigned char>>("clusTPCshmap");

  const auto& inputsTPCclusters = o2::tpc::getWorkflowTPCInput(pc);

  //
  MCLabelsTr lblITS;
  MCLabelsTr lblTPC;

  const MCLabelsCl* lblClusITSPtr = nullptr;
  std::unique_ptr<const MCLabelsCl> lblClusITS;

  if (mUseMC) {
    lblITS = pc.inputs().get<gsl::span<o2::MCCompLabel>>("trackITSMCTR");
    lblTPC = pc.inputs().get<gsl::span<o2::MCCompLabel>>("trackTPCMCTR");

    lblClusITS = pc.inputs().get<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("clusITSMCTR");
    lblClusITSPtr = lblClusITS.get();
  }
  //
  // create ITS clusters as spacepoints in tracking frame
  std::vector<o2::BaseCluster<float>> itsSP;
  itsSP.reserve(clusITS.size());
  auto pattIt = patterns.begin();
  o2::its::ioutils::convertCompactClusters(clusITS, pattIt, itsSP, mITSDict);

  // pass input data to MatchTPCITS object
  mMatching.setITSTracksInp(tracksITS);
  mMatching.setITSTrackClusIdxInp(trackClIdxITS);
  mMatching.setITSTrackROFRecInp(tracksITSROF);
  mMatching.setITSClustersInp(itsSP);
  mMatching.setITSClusterROFRecInp(clusITSROF);
  mMatching.setTPCTracksInp(tracksTPC);
  mMatching.setTPCTrackClusIdxInp(tracksTPCClRefs);
  mMatching.setTPCClustersInp(&inputsTPCclusters->clusterIndex);
  mMatching.setTPCClustersSharingMap(clusTPCShmap);

  if (mUseMC) {
    mMatching.setITSTrkLabelsInp(lblITS);
    mMatching.setTPCTrkLabelsInp(lblTPC);
    mMatching.setITSClsLabelsInp(lblClusITSPtr);
  }

  if (mUseFT0) {
    // Note: the particular variable will go out of scope, but the span is passed by copy to the
    // worker and the underlying memory is valid throughout the whole computation
    auto fitInfo = pc.inputs().get<gsl::span<o2::ft0::RecPoints>>("fitInfo");
    mMatching.setFITInfoInp(fitInfo);
  }

  const auto* dh = o2::header::get<o2::header::DataHeader*>(pc.inputs().get("trackITSROF").header);
  mMatching.setStartIR({0, dh->firstTForbit});

  //RSTODO: below is a hack, to remove once the framework will start propagating the header.firstTForbit
  if (tracksITSROF.size()) {
    mMatching.setStartIR(o2::raw::HBFUtils::Instance().getFirstIRofTF(tracksITSROF[0].getBCData()));
  }

  mMatching.run();

  pc.outputs().snapshot(Output{"GLO", "TPCITS", 0, Lifetime::Timeframe}, mMatching.getMatchedTracks());
  if (mUseMC) {
    pc.outputs().snapshot(Output{"GLO", "TPCITS_ITSMC", 0, Lifetime::Timeframe}, mMatching.getMatchedITSLabels());
    pc.outputs().snapshot(Output{"GLO", "TPCITS_TPCMC", 0, Lifetime::Timeframe}, mMatching.getMatchedTPCLabels());
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
  LOGF(INFO, "TPC-ITS matching total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getTPCITSMatchingSpec(bool useFT0, bool calib, bool useMC, const std::vector<int>& tpcClusLanes)
{

  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;
  inputs.emplace_back("trackITS", "ITS", "TRACKS", 0, Lifetime::Timeframe);
  inputs.emplace_back("trackClIdx", "ITS", "TRACKCLSID", 0, Lifetime::Timeframe);
  inputs.emplace_back("trackITSROF", "ITS", "ITSTrackROF", 0, Lifetime::Timeframe);
  inputs.emplace_back("clusITS", "ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("clusITSPatt", "ITS", "PATTERNS", 0, Lifetime::Timeframe);
  inputs.emplace_back("clusITSROF", "ITS", "CLUSTERSROF", 0, Lifetime::Timeframe);
  inputs.emplace_back("trackTPC", "TPC", "TRACKS", 0, Lifetime::Timeframe);
  inputs.emplace_back("trackTPCClRefs", "TPC", "CLUSREFS", 0, Lifetime::Timeframe);

  inputs.emplace_back("clusTPC", ConcreteDataTypeMatcher{"TPC", "CLUSTERNATIVE"}, Lifetime::Timeframe);
  inputs.emplace_back("clusTPCshmap", "TPC", "CLSHAREDMAP", 0, Lifetime::Timeframe);

  if (useFT0) {
    inputs.emplace_back("fitInfo", "FT0", "RECPOINTS", 0, Lifetime::Timeframe);
  }

  outputs.emplace_back("GLO", "TPCITS", 0, Lifetime::Timeframe);

  if (calib) {
    outputs.emplace_back("GLO", "TPCITS_VDHDTGL", 0, Lifetime::Timeframe);
  }

  if (useMC) {
    inputs.emplace_back("trackITSMCTR", "ITS", "TRACKSMCTR", 0, Lifetime::Timeframe);
    inputs.emplace_back("trackTPCMCTR", "TPC", "TRACKSMCLBL", 0, Lifetime::Timeframe);
    inputs.emplace_back("clusITSMCTR", "ITS", "CLUSTERSMCTR", 0, Lifetime::Timeframe);
    //
    outputs.emplace_back("GLO", "TPCITS_ITSMC", 0, Lifetime::Timeframe);
    outputs.emplace_back("GLO", "TPCITS_TPCMC", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "itstpc-track-matcher",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCITSMatchingDPL>(useFT0, calib, useMC)},
    Options{
      {"its-dictionary-path", VariantType::String, "", {"Path of the cluster-topology dictionary file"}},
      {"material-lut-path", VariantType::String, "", {"Path of the material LUT file"}},
      {"debug-tree-flags", VariantType::Int, 0, {"DebugFlagTypes bit-pattern for debug tree"}}}};
}

} // namespace globaltracking
} // namespace o2
