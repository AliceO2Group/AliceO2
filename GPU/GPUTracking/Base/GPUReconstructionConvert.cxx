// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionConvert.cxx
/// \author David Rohr

#include "GPUReconstructionConvert.h"
#include "TPCFastTransform.h"
#include "GPUTPCClusterData.h"
#include "ClusterNativeAccessExt.h"

using namespace GPUCA_NAMESPACE::gpu;

void GPUReconstructionConvert::ConvertNativeToClusterData(ClusterNativeAccessExt* native, std::unique_ptr<GPUTPCClusterData[]>* clusters, unsigned int* nClusters, const TPCFastTransform* transform, int continuousMaxTimeBin)
{
#ifdef HAVE_O2HEADERS
  memset(nClusters, 0, NSLICES * sizeof(nClusters[0]));
  unsigned int offset = 0;
  for (unsigned int i = 0; i < NSLICES; i++) {
    unsigned int nClSlice = 0;
    for (int j = 0; j < o2::TPC::Constants::MAXGLOBALPADROW; j++) {
      nClSlice += native->nClusters[i][j];
    }
    nClusters[i] = nClSlice;
    clusters[i].reset(new GPUTPCClusterData[nClSlice]);
    nClSlice = 0;
    for (int j = 0; j < o2::TPC::Constants::MAXGLOBALPADROW; j++) {
      for (unsigned int k = 0; k < native->nClusters[i][j]; k++) {
        const auto& cin = native->clusters[i][j][k];
        float x = 0, y = 0, z = 0;
        if (continuousMaxTimeBin == 0) {
          transform->Transform(i, j, cin.getPad(), cin.getTime(), x, y, z);
        } else {
          transform->TransformInTimeFrame(i, j, cin.getPad(), cin.getTime(), x, y, z, continuousMaxTimeBin);
        }
        auto& cout = clusters[i].get()[nClSlice];
        cout.x = x;
        cout.y = y;
        cout.z = z;
        cout.row = j;
        cout.amp = cin.qMax;
        cout.flags = cin.getFlags();
        cout.id = offset + k;
        nClSlice++;
      }
      native->clusterOffset[i][j] = offset;
      offset += native->nClusters[i][j];
    }
  }
#endif
}
