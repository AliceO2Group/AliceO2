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

#include "TTree.h"

#include "Framework/ControlService.h"
#include "GlobalTrackingWorkflow/TPCITSMatchingSpec.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
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

using namespace o2::framework;
using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

namespace o2
{
namespace globaltracking
{

void TPCITSMatchingDPL::init(InitContext& ic)
{

  //-------- init geometry and field --------//
  o2::base::GeometryManager::loadGeometry("O2geometry.root");
  o2::base::Propagator::initFieldFromGRP("o2sim_grp.root");

  mMatching.setDPLIO(true);
  const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
  mMatching.setITSROFrameLengthMUS(alpParams.roFrameLength / 1.e3); // ITS ROFrame duration in \mus
  mMatching.setMCTruthOn(mUseMC);
  //
  // configure matching cuts, should be eventually read from the CCDB
  mMatching.setCutMatchingChi2(100.);
  std::array<float, o2::track::kNParams> cutsAbs = { 2.f, 2.f, 0.2f, 0.2f, 4.f };
  std::array<float, o2::track::kNParams> cutsNSig2 = { 49.f, 49.f, 49.f, 49.f, 49.f };
  mMatching.setCrudeAbsDiffCut(cutsAbs);
  mMatching.setCrudeNSigma2Cut(cutsNSig2);
  mMatching.setTPCTimeEdgeZSafeMargin(3);

  mMatching.init();
}

void TPCITSMatchingDPL::run(ProcessingContext& pc)
{
  LOG(INFO) << "TPCITSMatchingDPL " << mFinished << " NLanes " << mTPCClusLanes.size();
  if (mFinished) {
    return;
  }

  auto tracksITS = pc.inputs().get<const std::vector<o2::its::TrackITS>>("trackITS");
  auto trackClIdxITS = pc.inputs().get<gsl::span<int>>("trackClIdx");
  auto tracksITSROF = pc.inputs().get<const std::vector<o2::itsmft::ROFRecord>>("trackITSROF");
  auto clusITS = pc.inputs().get<const std::vector<o2::itsmft::Cluster>>("clusITS");
  auto clusITSROF = pc.inputs().get<const std::vector<o2::itsmft::ROFRecord>>("clusITSROF");
  auto tracksTPC = pc.inputs().get<const std::vector<o2::tpc::TrackTPC>>("trackTPC");

  // TODO: this is very ugly way to transfer the vector to the matcher, we should pass a span
  std::vector<int> clIdx;
  clIdx.reserve(trackClIdxITS.size());
  std::copy(trackClIdxITS.begin(), trackClIdxITS.end(), std::inserter(clIdx, clIdx.end()));

  //---------------------------->> TPC Clusters loading >>------------------------------------------
  int operation = 0;
  uint64_t activeSectors = 0;
  std::bitset<o2::tpc::Constants::MAXSECTOR> validSectors = 0;
  std::map<int, DataRef> datarefs;
  for (auto const& lane : mTPCClusLanes) {
    std::string inputLabel = "clusTPC" + std::to_string(lane);
    LOG(INFO) << "Reading lane " << lane << " " << inputLabel;
    auto ref = pc.inputs().get(inputLabel);
    auto const* sectorHeader = DataRefUtils::getHeader<o2::tpc::TPCSectorHeader*>(ref);
    if (sectorHeader == nullptr) {
      // FIXME: think about error policy
      LOG(ERROR) << "sector header missing on header stack";
      return;
    }
    const int& sector = sectorHeader->sector;
    // check the current operation, this is used to either signal eod or noop
    // FIXME: the noop is not needed any more once the lane configuration with one
    // channel per sector is used
    if (sector < 0) {
      if (operation < 0 && operation != sector) {
        // we expect the same operation on all inputs
        LOG(ERROR) << "inconsistent lane operation, got " << sector << ", expecting " << operation;
      } else if (operation == 0) {
        // store the operation
        operation = sector;
      }
      continue;
    }
    if (validSectors.test(sector)) {
      // have already data for this sector, this should not happen in the current
      // sequential implementation, for parallel path merged at the tracker stage
      // multiple buffers need to be handled
      throw std::runtime_error("can only have one data set per sector");
    }
    activeSectors |= sectorHeader->activeSectors;
    validSectors.set(sector);
    datarefs[sector] = ref;
  }

  if (operation == -1) {
    // EOD is transmitted in the sectorHeader with sector number equal to -1
    LOG(WARNING) << "operation = " << operation;
  }

  auto printInputLog = [&validSectors, &activeSectors](auto& r, const char* comment, auto& s) {
    LOG(INFO) << comment << " " << *(r.spec) << ", size " << DataRefUtils::getPayloadSize(r) //
              << " for sector " << s                                                         //
              << std::endl                                                                   //
              << "  input status:   " << validSectors                                        //
              << std::endl                                                                   //
              << "  active sectors: " << std::bitset<o2::tpc::Constants::MAXSECTOR>(activeSectors);
  };

  if (activeSectors == 0 || (activeSectors & validSectors.to_ulong()) != activeSectors) {
    // not all sectors available
    // Since we expect complete input, this should not happen (why does the bufferization considered for TPC CA tracker? Ask Matthias)
    throw std::runtime_error("Did not receive TPC clusters data for all sectors");
    /*
    for (auto const& refentry : datarefs) {
      auto& sector = refentry.first;
      auto& ref = refentry.second;
      auto payploadSize = DataRefUtils::getPayloadSize(ref);
      mBufferedTPCClusters[sector].resize(payploadSize);
      std::copy(ref.payload, ref.payload + payploadSize, mBufferedTPCClusters[sector].begin());
      
      printInputLog(ref, "buffering", sector);
    }
    // not needed to send something, DPL will simply drop this timeslice, whenever the
    // data for all sectors is available, the output is sent in that time slice
    return;
    */
  }

  //------------------------------------------------------------------------------
  std::array<std::vector<MCLabelContainer>, o2::tpc::Constants::MAXSECTOR> mcInputs; // DUMMY
  std::array<gsl::span<const char>, o2::tpc::Constants::MAXSECTOR> clustersTPC;
  auto sectorStatus = validSectors;

  for (auto const& refentry : datarefs) {
    auto& sector = refentry.first;
    auto& ref = refentry.second;
    clustersTPC[sector] = gsl::span(ref.payload, DataRefUtils::getPayloadSize(ref));
    sectorStatus.reset(sector);
    printInputLog(ref, "received", sector);
  }
  if (sectorStatus.any()) {
    LOG(ERROR) << "Reading bufferized TPC clusters, this should not happen";
    // some of the inputs have been buffered
    for (size_t sector = 0; sector < sectorStatus.size(); ++sector) {
      if (sectorStatus.test(sector)) {
        clustersTPC[sector] = gsl::span(&mBufferedTPCClusters[sector].front(), mBufferedTPCClusters[sector].size());
      }
    }
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
  o2::tpc::MCLabelContainer clusterMCBuffer;
  memset(&clusterIndex, 0, sizeof(clusterIndex));
  o2::tpc::ClusterNativeHelper::Reader::fillIndex(clusterIndex, clusterBuffer, clusterMCBuffer, clustersTPC, mcInputs, [&validSectors](auto& index) { return validSectors.test(index); });

  //----------------------------<< TPC Clusters loading <<------------------------------------------

  //
  const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* lblITSPtr = nullptr;
  std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>> lblITS;
  const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* lblTPCPtr = nullptr;
  std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>> lblTPC;
  if (mUseMC) {
    lblITS = pc.inputs().get<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("trackITSMCTR");
    lblITSPtr = lblITS.get();
    lblTPC = pc.inputs().get<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("trackTPCMCTR");
    lblTPCPtr = lblTPC.get();
  }
  //
  // pass input data to MatchTPCITS object
  mMatching.setITSTracksInp(&tracksITS);
  mMatching.setITSTrackClusIdxInp(&clIdx);
  mMatching.setITSTrackROFRecInp(&tracksITSROF);
  mMatching.setITSClustersInp(&clusITS);
  mMatching.setITSClusterROFRecInp(&clusITSROF);
  mMatching.setTPCTracksInp(&tracksTPC);
  mMatching.setTPCClustersInp(&clusterIndex);

  if (mUseMC) {
    mMatching.setITSTrkLabelsInp(lblITSPtr);
    mMatching.setTPCTrkLabelsInp(lblTPCPtr);
  }

  const std::vector<o2::ft0::RecPoints>* rpFIT = nullptr;
  std::unique_ptr<const std::vector<o2::ft0::RecPoints>> rpFITU;
  if (mUseFIT) {
    rpFITU = pc.inputs().get<const std::vector<o2::ft0::RecPoints>*>("fitInfo");
    rpFIT = rpFITU.get();
    mMatching.setFITInfoInp(rpFIT);
  }

  mMatching.run();

  /* // at the moment we don't assume need for bufferization, no nead to clear
  for (auto& secClBuf : mBufferedTPCClusters) {
    secClBuf.clear();
  }
  */
  pc.outputs().snapshot(Output{ "GLO", "TPCITS", 0, Lifetime::Timeframe }, mMatching.getMatchedTracks());
  if (mUseMC) {
    pc.outputs().snapshot(Output{ "GLO", "TPCITS_ITSMC", 0, Lifetime::Timeframe }, mMatching.getMatchedITSLabels());
    pc.outputs().snapshot(Output{ "GLO", "TPCITS_TPCMC", 0, Lifetime::Timeframe }, mMatching.getMatchedTPCLabels());
  }
  mFinished = true;
  pc.services().get<ControlService>().readyToQuit(false);
}

DataProcessorSpec getTPCITSMatchingSpec(bool useMC, bool useFIT, const std::vector<int>& tpcClusLanes)
{

  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;
  inputs.emplace_back("trackITS", "ITS", "TRACKS", 0, Lifetime::Timeframe);
  inputs.emplace_back("trackClIdx", "ITS", "TRACKCLSID", 0, Lifetime::Timeframe);
  inputs.emplace_back("trackITSROF", "ITS", "ITSTrackROF", 0, Lifetime::Timeframe);
  inputs.emplace_back("clusITS", "ITS", "CLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("clusITSROF", "ITS", "ITSClusterROF", 0, Lifetime::Timeframe);
  inputs.emplace_back("trackTPC", "TPC", "TRACKS", 0, Lifetime::Timeframe);
  for (auto lane : tpcClusLanes) {
    std::string clusBind = "clusTPC" + std::to_string(lane);
    inputs.emplace_back(clusBind.c_str(), "TPC", "CLUSTERNATIVE", lane, Lifetime::Timeframe);
  }
  if (useFIT) {
    inputs.emplace_back("fitInfo", "FT0", "RECPOINTS", 0, Lifetime::Timeframe);
  }

  outputs.emplace_back("GLO", "TPCITS", 0, Lifetime::Timeframe);

  if (useMC) {
    inputs.emplace_back("trackITSMCTR", "ITS", "TRACKSMCTR", 0, Lifetime::Timeframe);
    inputs.emplace_back("trackTPCMCTR", "TPC", "TRACKSMCLBL", 0, Lifetime::Timeframe);
    //
    outputs.emplace_back("GLO", "TPCITS_ITSMC", 0, Lifetime::Timeframe);
    outputs.emplace_back("GLO", "TPCITS_TPCMC", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "itstpc-track-matcher",
    inputs,
    outputs,
    AlgorithmSpec{ adaptFromTask<TPCITSMatchingDPL>(useMC, useFIT, tpcClusLanes) },
    Options{}
  };
}

} // namespace globaltracking
} // namespace o2
