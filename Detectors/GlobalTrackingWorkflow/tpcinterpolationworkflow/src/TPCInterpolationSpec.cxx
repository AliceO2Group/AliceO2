// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file  TPCInterpolationSpec.cxx

#include <vector>

#include "Framework/InputRecordWalker.h"
#include "DataFormatsITS/TrackITS.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/ClusterNativeHelper.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "TPCInterpolationWorkflow/TPCInterpolationSpec.h"

using namespace o2::framework;

namespace o2
{
namespace tpc
{

void TPCInterpolationDPL::init(InitContext& ic)
{
  //-------- init geometry and field --------//
  o2::base::GeometryManager::loadGeometry();
  o2::base::Propagator::initFieldFromGRP("o2sim_grp.root");
  mTimer.Stop();
  mTimer.Reset();
  mInterpolation.init();
}

void TPCInterpolationDPL::run(ProcessingContext& pc)
{
  mTimer.Start(false);
  const auto tracksITS = pc.inputs().get<gsl::span<o2::its::TrackITS>>("trackITS");
  const auto tracksTPC = pc.inputs().get<gsl::span<o2::tpc::TrackTPC>>("trackTPC");
  const auto tracksITSTPC = pc.inputs().get<gsl::span<o2::dataformats::TrackTPCITS>>("match");
  const auto tracksTPCClRefs = pc.inputs().get<gsl::span<o2::tpc::TPCClRefElem>>("trackTPCClRefs");
  const auto trackMatchesTOF = pc.inputs().get<gsl::span<o2::dataformats::MatchInfoTOF>>("matchTOF"); // FIXME missing reader
  const auto clustersTOF = pc.inputs().get<gsl::span<o2::tof::Cluster>>("clustersTOF");

  // TPC Cluster loading part is copied from TPCITSMatchingSpec.cxx
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
    const int& sector = sectorHeader->sector();
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

  // pass input data to TrackInterpolation object
  mInterpolation.setITSTracksInp(tracksITS);
  mInterpolation.setTPCTracksInp(tracksTPC);
  mInterpolation.setTPCTrackClusIdxInp(tracksTPCClRefs);
  mInterpolation.setTPCClustersInp(&clusterIndex);
  mInterpolation.setTOFMatchesInp(trackMatchesTOF);
  mInterpolation.setITSTPCTrackMatchesInp(tracksITSTPC);
  mInterpolation.setTOFClustersInp(clustersTOF);

  if (mUseMC) {
    // possibly MC labels will be used to check filtering procedure performance before interpolation
    // not yet implemented
  }

  LOG(INFO) << "TPC Interpolation Workflow initialized. Start processing...";

  mInterpolation.process();

  pc.outputs().snapshot(Output{"GLO", "TPCINT_TRK", 0, Lifetime::Timeframe}, mInterpolation.getReferenceTracks());
  pc.outputs().snapshot(Output{"GLO", "TPCINT_RES", 0, Lifetime::Timeframe}, mInterpolation.getClusterResiduals());
  mTimer.Stop();
}

void TPCInterpolationDPL::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "TPC residuals extraction total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getTPCInterpolationSpec(bool useMC, const std::vector<int>& tpcClusLanes)
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;

  inputs.emplace_back("trackITS", "ITS", "TRACKS", 0, Lifetime::Timeframe);
  inputs.emplace_back("trackTPC", "TPC", "TRACKS", 0, Lifetime::Timeframe);
  inputs.emplace_back("trackTPCClRefs", "TPC", "CLUSREFS", 0, Lifetime::Timeframe);

  inputs.emplace_back("clusTPC", ConcreteDataTypeMatcher{"TPC", "CLUSTERNATIVE"}, Lifetime::Timeframe);

  inputs.emplace_back("match", "GLO", "TPCITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("matchTOF", "TOF", "MATCHINFOS", 0, Lifetime::Timeframe);
  inputs.emplace_back("clustersTOF", "TOF", "CLUSTERS", 0, Lifetime::Timeframe);

  if (useMC) {
    LOG(FATAL) << "MC usage must be disabled for this workflow, since it is not yet implemented";
    // are the MC inputs from ITS-TPC matching and TOF matching duplicates? if trackITSMCTR == matchTOFMCITS one of them should be removed
    inputs.emplace_back("trackITSMCTR", "GLO", "TPCITS_ITSMC", 0, Lifetime::Timeframe);
    inputs.emplace_back("trackTPCMCTR", "GLO", "TPCITS_TPCMC", 0, Lifetime::Timeframe);
    inputs.emplace_back("matchTOFMC", o2::header::gDataOriginTOF, "MATCHTOFINFOSMC", 0, Lifetime::Timeframe);
    inputs.emplace_back("matchTOFMCTPC", o2::header::gDataOriginTOF, "MATCHTPCINFOSMC", 0, Lifetime::Timeframe);
    inputs.emplace_back("matchTOFMCITS", o2::header::gDataOriginTOF, "MATCHITSINFOSMC", 0, Lifetime::Timeframe);
  }

  outputs.emplace_back("GLO", "TPCINT_TRK", 0, Lifetime::Timeframe);
  outputs.emplace_back("GLO", "TPCINT_RES", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "tpc-track-interpolation",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCInterpolationDPL>(useMC)},
    Options{}};
}

} // namespace tpc
} // namespace o2
