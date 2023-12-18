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

/// \file GPUTPCDecompressionKernels.cxx
/// \author Gabriele Cimador

#include "GPUTPCDecompressionKernels.h"
#include "GPULogging.h"
#include "GPUConstantMem.h"


using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;

template <>
GPUdii() void GPUTPCDecompressionKernels::Thread<GPUTPCDecompressionKernels::test>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() GPUSharedMemory& smem, processorType& processors)
{
  GPUTPCDecompression& GPUrestrict() decompressor = processors.tpcDecompressor;
  CompressedClusters& GPUrestrict() cmprClusters = decompressor.mInputGPU;
  const GPUParam& GPUrestrict() param = processors.param;

  unsigned int offset = 0, lasti = 0;
  const unsigned int maxTime = (param.par.continuousMaxTimeBin + 1) * ClusterNative::scaleTimePacked - 1;

  for (unsigned int i = get_global_id(0); i < cmprClusters.nTracks; i += get_global_size(0)) {
    while (lasti < i) {
      offset += cmprClusters.nTrackClusters[lasti++];
    }
    lasti++;
    //decompressTrack(clustersCompressed, param, maxTime, i, offset, clusters, locks);

  }
  if (!iThread && !iBlock) {
    GPUInfo("==== on GPU nAttCl = %d, nUnAttCl = %d, nTracks = %d",cmprClusters.nAttachedClusters,cmprClusters.nUnattachedClusters,cmprClusters.nTracks);
    GPUInfo("=== sizeof(CluserNative) = %lu", sizeof(ClusterNative));
    /*int * test = new int[10];
    test[0] = 1;
    GPUInfo("==== got it %p -- %d",(void*)test,test[0]);
    delete[] test;*/
  }

}