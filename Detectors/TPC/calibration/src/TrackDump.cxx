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

#include <vector>
#include "GPUO2Interface.h"
#include "GPUDefOpenCL12Templates.h"
#include "GPUDefConstantsAndSettings.h"
#include "SliceTracker/GPUTPCGeometry.h"

#include "DataFormatsTPC/Defs.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "TPCBase/Mapper.h"
#include "TPCCalibration/TrackDump.h"

using namespace o2::tpc;
using namespace o2::tpc::constants;

void TrackDump::filter(const gsl::span<const TrackTPC> tracks, ClusterNativeAccess const& clusterIndex, const gsl::span<const o2::tpc::TPCClRefElem> clRefs, const gsl::span<const o2::MCCompLabel> mcLabels)
{
  if (!mTreeDump && outputFileName.size()) {
    mTreeDump = std::make_unique<utils::TreeStreamRedirector>(outputFileName.data(), "recreate");
  }

  std::vector<TrackInfo> tpcTracks;
  std::vector<std::vector<ClusterGlobal>> clustersGlobalEvent;
  std::vector<ClusterGlobal>* clustersGlobal{};
  std::vector<o2::MCCompLabel> tpcTracksMCTruth;

  GPUCA_NAMESPACE::gpu::GPUTPCGeometry gpuGeom;

  for (const auto& track : tracks) {
    const int nCl = track.getNClusterReferences();
    auto& trackInfo = tpcTracks.emplace_back(track);
    auto& clInfos = trackInfo.clusters;
    if (writeGlobal) {
      clustersGlobal = &clustersGlobalEvent.emplace_back();
    }

    for (int j = nCl - 1; j >= 0; j--) {
      uint8_t sector, padrow;
      uint32_t clusterIndexInRow;
      track.getClusterReference(clRefs, j, sector, padrow, clusterIndexInRow);
      const auto& cl = clusterIndex.clusters[sector][padrow][clusterIndexInRow];
      auto& clInfo = clInfos.emplace_back(cl);
      clInfo.sector = sector;
      clInfo.padrow = padrow;

      if (clustersGlobal) {
        auto& clGlobal = clustersGlobal->emplace_back(ClusterGlobal{clInfo.getGx(), clInfo.getGy(), cl.qMax, cl.qTot, sector, padrow});
      }
    }
  }
  if (writeMC) {
    for (const auto& mcLabel : mcLabels) {
      tpcTracksMCTruth.emplace_back(mcLabel);
    }
  }

  if (mTreeDump) {
    auto& tree = (*mTreeDump) << "tpcrec";
    if (writeTracks) {
      //  << "info=" << trackInfos
      tree << "TPCTracks=" << tpcTracks;
    }
    if (writeGlobal) {
      tree << "cls" << clustersGlobalEvent;
    }
    if (writeMC) {
      tree << "TPCTracksMCTruth=" << tpcTracksMCTruth;
    }
    tree << "\n";
    //  << "clusters=" << clInfoVec
  }
}

void TrackDump::finalize()
{
  if (mTreeDump) {
    mTreeDump->Close();
  }

  mTreeDump.reset();
}

float TrackDump::ClusterNativeAdd::getLx() const
{
  static GPUCA_NAMESPACE::gpu::GPUTPCGeometry gpuGeom;
  return gpuGeom.Row2X(padrow);
}

float TrackDump::ClusterNativeAdd::getLy() const
{
  static GPUCA_NAMESPACE::gpu::GPUTPCGeometry gpuGeom;
  return gpuGeom.LinearPad2Y(sector, padrow, getPad());
}

float TrackDump::ClusterNativeAdd::getGx() const
{
  const LocalPosition2D l2D{getLx(), getLy()};
  const auto g2D = Mapper::LocalToGlobal(l2D, Sector(sector));
  return g2D.x();
}

float TrackDump::ClusterNativeAdd::getGy() const
{
  const LocalPosition2D l2D{getLx(), getLy()};
  const auto g2D = Mapper::LocalToGlobal(l2D, Sector(sector));
  return g2D.y();
}