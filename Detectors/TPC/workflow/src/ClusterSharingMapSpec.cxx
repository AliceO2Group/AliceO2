// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   ClusterSharingMapSpec.cxx
/// @brief Device to produce TPC clusters sharing map
/// \author ruben.shahoyan@cern.ch

#include <gsl/span>
#include <TStopwatch.h>
#include <vector>
#include "Framework/InputRecordWalker.h"
#include "DataFormatsTPC/ClusterNativeHelper.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "GPUO2InterfaceRefit.h"
#include "TPCWorkflow/ClusterSharingMapSpec.h"

using namespace o2::framework;
using namespace o2::tpc;

void ClusterSharingMapSpec::run(ProcessingContext& pc)
{
  TStopwatch timer;

  const auto tracksTPC = pc.inputs().get<gsl::span<o2::tpc::TrackTPC>>("trackTPC");
  const auto tracksTPCClRefs = pc.inputs().get<gsl::span<o2::tpc::TPCClRefElem>>("trackTPCClRefs");

  //---------------------------->> TPC Clusters loading >>------------------------------------------
  // FIXME: this is a copy-paste from TPCITSMatchingSpec version of TPC cluster reading, later to be organized in a better way
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

  auto& bufVec = pc.outputs().make<std::vector<unsigned char>>(Output{o2::header::gDataOriginTPC, "CLSHAREDMAP", 0}, clusterIndex.nClustersTotal);
  o2::gpu::GPUTPCO2InterfaceRefit::fillSharedClustersMap(&clusterIndex, tracksTPC, tracksTPCClRefs.data(), bufVec.data());

  timer.Stop();
  LOGF(INFO, "Timing for TPC clusters sharing map creation: Cpu: %.3e Real: %.3e s", timer.CpuTime(), timer.RealTime());
}
