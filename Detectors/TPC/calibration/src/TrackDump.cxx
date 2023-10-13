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

#include <memory>
#include <vector>
#include <filesystem>
#include <fmt/format.h>
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
namespace fs = std::filesystem;

o2::gpu::CorrectionMapsHelper o2::tpc::TrackDump::ClusterNativeAdd::sCorrHelper{};

void TrackDump::filter(const gsl::span<const TrackTPC> tracks, ClusterNativeAccess const& clusterIndex, const gsl::span<const o2::tpc::TPCClRefElem> clRefs, const gsl::span<const o2::MCCompLabel> mcLabels)
{
  if (!mTreeDump && outputFileName.size()) {
    mTreeDump = std::make_unique<utils::TreeStreamRedirector>(outputFileName.data(), "recreate");
    const auto fspath = fs::path(outputFileName);
    const std::string path = fspath.parent_path().c_str();
    const std::string baseName = fspath.stem().c_str();
    if (clusterStorageType == ClStorageType::SeparateFile || noTrackClusterType == ClUnStorageType::SeparateFile) {
      mTreeDumpClOnly = std::make_unique<utils::TreeStreamRedirector>(fmt::format("{}/{}.cl.root", path, baseName).data(), "recreate");
    }
  }

  std::vector<TrackInfo> tpcTrackInfos;
  std::vector<TrackTPC> tpcTracks;
  std::vector<std::vector<ClusterNativeAdd>> clusters;
  std::vector<std::vector<ClusterGlobal>> clustersGlobalEvent;
  std::vector<ClusterGlobal>* clustersGlobal{};
  std::vector<o2::MCCompLabel> tpcTracksMCTruth;

  ClExcludes excludes;

  const GPUCA_NAMESPACE::gpu::GPUTPCGeometry gpuGeom;

  for (const auto& track : tracks) {
    const int nCl = track.getNClusterReferences();
    std::vector<ClusterNativeAdd>* clustersPtr{nullptr};
    if (clusterStorageType == ClStorageType::InsideTrack) {
      auto& trackInfo = tpcTrackInfos.emplace_back(track);
      clustersPtr = &trackInfo.clusters;
    } else {
      tpcTracks.emplace_back(track);
      clustersPtr = &clusters.emplace_back();
    }
    if (writeGlobal) {
      clustersGlobal = &clustersGlobalEvent.emplace_back();
    }
    auto& clInfos = *clustersPtr;

    for (int j = nCl - 1; j >= 0; j--) {
      uint8_t sector, padrow;
      uint32_t clusterIndexInRow;
      track.getClusterReference(clRefs, j, sector, padrow, clusterIndexInRow);
      const auto& cl = clusterIndex.clusters[sector][padrow][clusterIndexInRow];
      auto& clInfo = clInfos.emplace_back(cl);
      clInfo.sector = sector;
      clInfo.padrow = padrow;
      excludes[sector][padrow].emplace_back(clusterIndexInRow);

      if (clustersGlobal) {
        auto& clGlobal = clustersGlobal->emplace_back(ClusterGlobal{clInfo.gx(), clInfo.gy(), cl.qMax, cl.qTot, sector, padrow});
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
      if (clusterStorageType == ClStorageType::InsideTrack) {
        tree << "TPCTracks=" << tpcTrackInfos;
      } else {
        tree << "TPCTracks=" << tpcTracks;
        if (clusterStorageType == ClStorageType::SeparateBranch) {
          tree << "cls=" << clusters;
        } else if (clusterStorageType == ClStorageType::SeparateTree) {
          auto& cltree = (*mTreeDump) << "cls";
          cltree << "cls=" << clusters
                 << "\n";
        }
      }
    }
    if (writeGlobal) {
      tree << "clsGlo=" << clustersGlobalEvent;
    }
    if (writeMC) {
      tree << "TPCTracksMCTruth=" << tpcTracksMCTruth;
    }
    tree << "\n";
    //  << "clusters=" << clInfoVec
  }

  // ===| clusters not associated to tracks |===================================
  utils::TreeStreamRedirector* treeDumpUn = nullptr;

  if (noTrackClusterType == ClUnStorageType::SeparateTree && mTreeDump) {
    treeDumpUn = mTreeDump.get();
  } else if (noTrackClusterType == ClUnStorageType::SeparateFile && mTreeDumpClOnly) {
    treeDumpUn = mTreeDumpClOnly.get();
  }
  if (treeDumpUn) {
    std::vector<ClusterNativeAdd> clustersUn;
    fillClNativeAdd(clusterIndex, clustersUn, &excludes);
    auto& cltree = (*treeDumpUn) << "clsn";
    cltree << "cls=" << clustersUn
           << "\n";
  }
}

void TrackDump::finalize()
{
  if (mTreeDump) {
    mTreeDump->Close();
  }

  mTreeDump.reset();
}

void TrackDump::fillClNativeAdd(ClusterNativeAccess const& clusterIndex, std::vector<ClusterNativeAdd>& clInfos, ClExcludes* excludes)
{
  const GPUCA_NAMESPACE::gpu::GPUTPCGeometry gpuGeom;

  for (int sector = 0; sector < MAXSECTOR; ++sector) {
    for (int padrow = 0; padrow < MAXGLOBALPADROW; ++padrow) {

      for (size_t icl = 0; icl < clusterIndex.nClusters[sector][padrow]; ++icl) {
        if (excludes) {
          const auto& exRow = (*excludes)[sector][padrow];
          if (std::find(exRow.begin(), exRow.end(), icl) != exRow.end()) {
            continue;
          }
        }
        const auto& cl = clusterIndex.clusters[sector][padrow][icl];
        auto& clInfo = clInfos.emplace_back(cl);
        clInfo.sector = sector;
        clInfo.padrow = padrow;
      }
    }
  }
}

float TrackDump::ClusterNativeAdd::cpad() const
{
  const GPUCA_NAMESPACE::gpu::GPUTPCGeometry gpuGeom;
  return getPad() - gpuGeom.NPads(padrow) / 2.f;
}

float TrackDump::ClusterNativeAdd::lx() const
{
  const GPUCA_NAMESPACE::gpu::GPUTPCGeometry gpuGeom;
  return gpuGeom.Row2X(padrow);
}

float TrackDump::ClusterNativeAdd::ly() const
{
  const GPUCA_NAMESPACE::gpu::GPUTPCGeometry gpuGeom;
  return gpuGeom.LinearPad2Y(sector, padrow, getPad());
}

float TrackDump::ClusterNativeAdd::gx() const
{
  const LocalPosition2D l2D{lx(), ly()};
  const auto g2D = Mapper::LocalToGlobal(l2D, Sector(sector));
  return g2D.x();
}

float TrackDump::ClusterNativeAdd::gy() const
{
  const LocalPosition2D l2D{lx(), ly()};
  const auto g2D = Mapper::LocalToGlobal(l2D, Sector(sector));
  return g2D.y();
}

float TrackDump::ClusterNativeAdd::lxc(float vertexTime) const
{
  float x{0.f}, y{0.f}, z{0.f};
  if (sCorrHelper.getCorrMap()) {
    sCorrHelper.Transform(sector, padrow, getPad(), getTime(), x, y, z, vertexTime);
  }
  return x;
}

float TrackDump::ClusterNativeAdd::lyc(float vertexTime) const
{
  float x{0.f}, y{0.f}, z{0.f};
  if (sCorrHelper.getCorrMap()) {
    sCorrHelper.Transform(sector, padrow, getPad(), getTime(), x, y, z, vertexTime);
  }
  return y;
}

float TrackDump::ClusterNativeAdd::gxc(float vertexTime) const
{
  const LocalPosition2D l2D{lxc(), lyc()};
  const auto g2D = Mapper::LocalToGlobal(l2D, Sector(sector));
  return g2D.x();
}

float TrackDump::ClusterNativeAdd::gyc(float vertexTime) const
{
  const LocalPosition2D l2D{lxc(), lyc()};
  const auto g2D = Mapper::LocalToGlobal(l2D, Sector(sector));
  return g2D.y();
}

float TrackDump::ClusterNativeAdd::zc(float vertexTime) const
{
  float x{0.f}, y{0.f}, z{0.f};
  if (sCorrHelper.getCorrMap()) {
    sCorrHelper.Transform(sector, padrow, getPad(), getTime(), x, y, z, vertexTime);
  }
  return z;
}

void TrackDump::ClusterNativeAdd::loadCorrMaps(std::string_view corrMapFile, std::string_view corrMapFileRef)
{
  sCorrHelper.setOwner(true);
  sCorrHelper.setCorrMap(gpu::TPCFastTransform::loadFromFile(corrMapFile.data()));
  if (!corrMapFileRef.empty()) {
    sCorrHelper.setCorrMapRef(gpu::TPCFastTransform::loadFromFile(corrMapFileRef.data()));
  }
}
