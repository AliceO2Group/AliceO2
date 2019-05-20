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
#include "ClusterNativeAccessExt.h"
#include <algorithm>
#include <cstring>

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::TPC;

int TPCClusterDecompressor::decompress(const CompressedClusters* clustersCompressed, o2::TPC::ClusterNativeAccessFullTPC& clustersNative, std::vector<o2::TPC::ClusterNative>& clusterBuffer)
{
  std::vector<ClusterNative> clusters[NSLICES][GPUCA_ROW_COUNT];
  unsigned int offset = 0;
  for (unsigned int i = 0; i < clustersCompressed->nTracks; i++) {
    unsigned int slice = clustersCompressed->sliceA[i];
    unsigned int row = clustersCompressed->rowA[i];
    for (unsigned int j = 0; j < clustersCompressed->nTrackClusters[i]; j++) {
      unsigned int pad = 0, time = 0;
      if (j) {
        slice = clustersCompressed->sliceLegDiffA[offset - i - 1];
        if (slice > NSLICES) {
          slice -= NSLICES;
        }
        row = clustersCompressed->rowDiffA[offset - i - 1];
        time = clustersCompressed->timeResA[offset - i - 1];
        pad = clustersCompressed->padResA[offset - i - 1];
      } else {
        time = clustersCompressed->timeA[i];
        pad = clustersCompressed->padA[i];
      }
      clusters[slice][row].emplace_back(time, clustersCompressed->flagsA[offset], pad, clustersCompressed->sigmaTimeA[offset], clustersCompressed->sigmaPadA[offset], clustersCompressed->qMaxA[offset], clustersCompressed->qTotA[offset]);
      offset++;
    }
  }
  clusterBuffer.resize(clustersCompressed->nAttachedClusters + clustersCompressed->nUnattachedClusters);
  offset = 0;
  unsigned int offset2 = 0;
  for (unsigned int i = 0; i < NSLICES; i++) {
    for (unsigned int j = 0; j < GPUCA_ROW_COUNT; j++) {
      ClusterNative* ptr = &clusterBuffer[offset];
      clustersNative.clusters[i][j] = ptr;
      memcpy((void*)&clusterBuffer[offset], (const void*)clusters[i][j].data(), clusters[i][j].size() * sizeof(clusterBuffer[0]));
      offset += clusters[i][j].size();
      for (unsigned int k = 0; k < clustersCompressed->nSliceRowClusters[i * GPUCA_ROW_COUNT + j]; k++) {
        clusterBuffer[offset] = ClusterNative(clustersCompressed->timeDiffU[offset2], clustersCompressed->flagsU[offset2], clustersCompressed->padDiffU[offset2], clustersCompressed->sigmaTimeU[offset2], clustersCompressed->sigmaPadU[offset2], clustersCompressed->qMaxU[offset2],
                                              clustersCompressed->qTotU[offset2]);
        offset++;
        offset2++;
      }
      clustersNative.nClusters[i][j] = clusters[i][j].size() + clustersCompressed->nSliceRowClusters[i * GPUCA_ROW_COUNT + j];
      std::sort(ptr, ptr + clustersNative.nClusters[i][j]);
    }
  }

  return 0;
}
