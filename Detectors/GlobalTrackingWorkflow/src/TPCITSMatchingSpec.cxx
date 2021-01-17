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
#include "Framework/InputRecordWalker.h"
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
#include "DataFormatsTPC/ClusterNativeHelper.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
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

  //---------------------------->> TPC Clusters loading >>------------------------------------------
  int operation = 0;
  uint64_t activeSectors = 0;
  std::bitset<o2::tpc::constants::MAXSECTOR> validSectors = 0;
  std::map<int, DataRef> datarefs;
  std::vector<InputSpec> filter = {
    {"check", ConcreteDataTypeMatcher{"TPC", "CLUSTERNATIVE"}, Lifetime::Timeframe},
  };
  for (auto const& ref : InputRecordWalker(pc.inputs(), filter)) {
    auto const* sectorHeader = DataRefUtils::getHeader<o2::tpc::TPCSectorHeader*>(ref);
    if (sectorHeader == nullptr) {
      // FIXME: think about error policy
      LOG(ERROR) << "sector header missing on header stack";
      throw std::runtime_error("sector header missing on header stack");
    }
    int sector = sectorHeader->sector();
    std::bitset<o2::tpc::constants::MAXSECTOR> sectorMask(sectorHeader->sectorBits);
    LOG(INFO) << "Reading TPC cluster data, sector mask is " << sectorMask;
    if ((validSectors & sectorMask).any()) {
      // have already data for this sector, this should not happen in the current
      // sequential implementation, for parallel path merged at the tracker stage
      // multiple buffers need to be handled
      throw std::runtime_error("can only have one data set per sector");
    }
    activeSectors |= sectorHeader->activeSectors;
    validSectors |= sectorMask;
    datarefs[sector] = ref;
  }

  auto printInputLog = [&validSectors, &activeSectors](auto& r, const char* comment, auto& s) {
    LOG(INFO) << comment << " " << *(r.spec) << ", size " << DataRefUtils::getPayloadSize(r) //
              << " for sector " << s                                                         //
              << std::endl                                                                   //
              << "  input status:   " << validSectors                                        //
              << std::endl                                                                   //
              << "  active sectors: " << std::bitset<o2::tpc::constants::MAXSECTOR>(activeSectors);
  };

  if (activeSectors == 0 || (activeSectors & validSectors.to_ulong()) != activeSectors) {
    // not all sectors available
    // Since we expect complete input, this should not happen (why does the bufferization considered for TPC CA tracker? Ask Matthias)
    throw std::runtime_error("Did not receive TPC clusters data for all sectors");
  }
  //------------------------------------------------------------------------------
  std::vector<gsl::span<const char>> clustersTPC;

  for (auto const& refentry : datarefs) {
    auto& sector = refentry.first;
    auto& ref = refentry.second;
    clustersTPC.emplace_back(ref.payload, DataRefUtils::getPayloadSize(ref));
    printInputLog(ref, "received", sector);
  }

  // Just print TPC clusters status
  {
    // make human readable information from the bitfield
    std::string bitInfo;
    auto nActiveBits = validSectors.count();
    if (((uint64_t)0x1 << nActiveBits) == validSectors.to_ulong() + 1) {
      // sectors 0 to some upper bound are active
      bitInfo = "0-" + std::to_string(nActiveBits - 1);
    } else {
      int rangeStart = -1;
      int rangeEnd = -1;
      for (size_t sector = 0; sector < validSectors.size(); sector++) {
        if (validSectors.test(sector)) {
          if (rangeStart < 0) {
            if (rangeEnd >= 0) {
              bitInfo += ",";
            }
            bitInfo += std::to_string(sector);
            if (nActiveBits == 1) {
              break;
            }
            rangeStart = sector;
          }
          rangeEnd = sector;
        } else {
          if (rangeStart >= 0 && rangeEnd > rangeStart) {
            bitInfo += "-" + std::to_string(rangeEnd);
          }
          rangeStart = -1;
        }
      }
      if (rangeStart >= 0 && rangeEnd > rangeStart) {
        bitInfo += "-" + std::to_string(rangeEnd);
      }
    }
    LOG(INFO) << "running matching for sector(s) " << bitInfo;
  }

  o2::tpc::ClusterNativeAccess clusterIndex;
  std::unique_ptr<o2::tpc::ClusterNative[]> clusterBuffer;
  memset(&clusterIndex, 0, sizeof(clusterIndex));
  o2::tpc::ClusterNativeHelper::ConstMCLabelContainerViewWithBuffer dummyMCOutput;
  std::vector<o2::tpc::ClusterNativeHelper::ConstMCLabelContainerView> dummyMCInput;
  o2::tpc::ClusterNativeHelper::Reader::fillIndex(clusterIndex, clusterBuffer, dummyMCOutput, clustersTPC, dummyMCInput);
  //----------------------------<< TPC Clusters loading <<------------------------------------------

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
  mMatching.setTPCClustersInp(&clusterIndex);
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
