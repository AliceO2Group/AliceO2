// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TPCClusterDecompressor.cxx
/// \author David Rohr

#include "TPCClusterDecompressor.h"
#include "GPUO2DataTypes.h"
#include "GPUParam.h"
#include "GPUTPCCompressionTrackModel.h"
#include <algorithm>
#include <cstring>

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;

int TPCClusterDecompressor::decompress(const CompressedClustersFlat* clustersCompressed, o2::tpc::ClusterNativeAccess& clustersNative, std::function<o2::tpc::ClusterNative*(size_t)> allocator, const GPUParam& param)
{
  CompressedClusters c = *clustersCompressed;
  return decompress(&c, clustersNative, allocator, param);
}

int TPCClusterDecompressor::decompress(const CompressedClusters* clustersCompressed, o2::tpc::ClusterNativeAccess& clustersNative, std::function<o2::tpc::ClusterNative*(size_t)> allocator, const GPUParam& param)
{
  std::vector<ClusterNative> clusters[NSLICES][GPUCA_ROW_COUNT];
  unsigned int offset = 0;
  for (unsigned int i = 0; i < clustersCompressed->nTracks; i++) {
    unsigned int slice = clustersCompressed->sliceA[i];
    unsigned int row = clustersCompressed->rowA[i];
    GPUTPCCompressionTrackModel track;
    for (unsigned int j = 0; j < clustersCompressed->nTrackClusters[i]; j++) {
      unsigned int pad = 0, time = 0;
      if (j) {
        unsigned char tmpSlice = clustersCompressed->sliceLegDiffA[offset - i - 1];
        bool changeLeg = (tmpSlice >= NSLICES);
        if (changeLeg) {
          tmpSlice -= NSLICES;
        }
        if (clustersCompressed->nComppressionModes & GPUSettings::CompressionDifferences) {
          slice += tmpSlice;
          if (slice >= NSLICES) {
            slice -= NSLICES;
          }
          row += clustersCompressed->rowDiffA[offset - i - 1];
          if (row >= GPUCA_ROW_COUNT) {
            row -= GPUCA_ROW_COUNT;
          }
        } else {
          slice = tmpSlice;
          row = clustersCompressed->rowDiffA[offset - i - 1];
        }
        if (changeLeg && track.Mirror()) {
          offset += clustersCompressed->nTrackClusters[i] - j;
          break;
        }
        if (track.Propagate(param.tpcGeometry.Row2X(row), param.SliceParam[slice].Alpha)) {
          offset += clustersCompressed->nTrackClusters[i] - j;
          break;
        }
        unsigned int timeTmp = clustersCompressed->timeResA[offset - i - 1];
        if (timeTmp & 800000) {
          timeTmp |= 0xFF000000;
        }
        time = timeTmp + ClusterNative::packTime(CAMath::Max(0.f, param.tpcGeometry.LinearZ2Time(slice, track.Z())));
        float tmpPad = CAMath::Max(0.f, CAMath::Min((float)param.tpcGeometry.NPads(GPUCA_ROW_COUNT - 1), param.tpcGeometry.LinearY2Pad(slice, row, track.Y())));
        pad = clustersCompressed->padResA[offset - i - 1] + ClusterNative::packPad(tmpPad);
      } else {
        time = clustersCompressed->timeA[i];
        pad = clustersCompressed->padA[i];
      }
      std::vector<ClusterNative>& clusterVector = clusters[slice][row];
      clusterVector.emplace_back(time, clustersCompressed->flagsA[offset], pad, clustersCompressed->sigmaTimeA[offset], clustersCompressed->sigmaPadA[offset], clustersCompressed->qMaxA[offset], clustersCompressed->qTotA[offset]);
      float y = param.tpcGeometry.LinearPad2Y(slice, row, clusterVector.back().getPad());
      float z = param.tpcGeometry.LinearTime2Z(slice, clusterVector.back().getTime());
      if (j == 0) {
        track.Init(param.tpcGeometry.Row2X(row), y, z, param.SliceParam[slice].Alpha, clustersCompressed->qPtA[i], param);
      }
      if (j + 1 < clustersCompressed->nTrackClusters[i] && track.Filter(y, z, row)) {
        offset += clustersCompressed->nTrackClusters[i] - j;
        break;
      }
      offset++;
    }
  }
  ClusterNative* clusterBuffer = allocator(clustersCompressed->nAttachedClusters + clustersCompressed->nUnattachedClusters);
  unsigned int offsets[NSLICES][GPUCA_ROW_COUNT];
  offset = 0;
  for (unsigned int i = 0; i < NSLICES; i++) {
    for (unsigned int j = 0; j < GPUCA_ROW_COUNT; j++) {
      clustersNative.nClusters[i][j] = clusters[i][j].size() + clustersCompressed->nSliceRowClusters[i * GPUCA_ROW_COUNT + j];
      offsets[i][j] = offset;
      offset += clustersCompressed->nSliceRowClusters[i * GPUCA_ROW_COUNT + j];
    }
  }
  clustersNative.clustersLinear = clusterBuffer;
  clustersNative.setOffsetPtrs();
  GPUCA_OPENMP(parallel for)
  for (unsigned int i = 0; i < NSLICES; i++) {
    for (unsigned int j = 0; j < GPUCA_ROW_COUNT; j++) {
      ClusterNative* buffer = &clusterBuffer[clustersNative.clusterOffset[i][j]];
      if (clusters[i][j].size()) {
        memcpy((void*)buffer, (const void*)clusters[i][j].data(), clusters[i][j].size() * sizeof(clusterBuffer[0]));
      }
      unsigned int time = 0;
      unsigned short pad = 0;
      ClusterNative* cl = buffer + clusters[i][j].size();
      unsigned int end = offsets[i][j] + clustersCompressed->nSliceRowClusters[i * GPUCA_ROW_COUNT + j];
      for (unsigned int k = offsets[i][j]; k < end; k++) {
        if (clustersCompressed->nComppressionModes & GPUSettings::CompressionDifferences) {
          unsigned int timeTmp = clustersCompressed->timeDiffU[k];
          if (timeTmp & 800000) {
            timeTmp |= 0xFF000000;
          }
          time += timeTmp;
          pad += clustersCompressed->padDiffU[k];
        } else {
          time = clustersCompressed->timeDiffU[k];
          pad = clustersCompressed->padDiffU[k];
        }
        *(cl++) = ClusterNative(time, clustersCompressed->flagsU[k], pad, clustersCompressed->sigmaTimeU[k], clustersCompressed->sigmaPadU[k], clustersCompressed->qMaxU[k], clustersCompressed->qTotU[k]);
      }
      std::sort(buffer, buffer + clustersNative.nClusters[i][j]);
    }
  }

  return 0;
}
