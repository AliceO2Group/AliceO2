// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CompressedClusters.h
/// \brief Container to store compressed TPC cluster data
/// \author David Rohr

#ifndef ALICEO2_DATAFORMATSTPC_COMPRESSED_CLUSTERS_H
#define ALICEO2_DATAFORMATSTPC_COMPRESSED_CLUSTERS_H

#include "GPUCommonRtypes.h"

namespace o2
{
namespace TPC
{
struct CompressedClustersCounters {
  unsigned int nTracks = 0;
  unsigned int nAttachedClusters = 0;
  unsigned int nUnattachedClusters = 0;
  unsigned int nAttachedClustersReduced = 0;
  unsigned int nSliceRows = 36 * 152;
  ClassDefNV(CompressedClustersCounters, 1);
};

template <class T>
struct CompressedClustersPtrs_helper : public T {
  unsigned short* qTotA = nullptr;        //[nAttachedClusters]
  unsigned short* qMaxA = nullptr;        //[nAttachedClusters]
  unsigned char* flagsA = nullptr;        //[nAttachedClusters]
  unsigned char* rowDiffA = nullptr;      //[nAttachedClustersReduced]
  unsigned char* sliceLegDiffA = nullptr; //[nAttachedClustersReduced]
  unsigned short* padResA = nullptr;      //[nAttachedClustersReduced]
  unsigned int* timeResA = nullptr;       //[nAttachedClustersReduced]
  unsigned char* sigmaPadA = nullptr;     //[nAttachedClusters]
  unsigned char* sigmaTimeA = nullptr;    //[nAttachedClusters]

  char* qPtA = nullptr;            //[nTracks]
  unsigned char* rowA = nullptr;   //[nTracks]
  unsigned char* sliceA = nullptr; //[nTracks]
  unsigned int* timeA = nullptr;   //[nTracks]
  unsigned short* padA = nullptr;  //[nTracks]

  unsigned short* qTotU = nullptr;     //[nUnattachedClusters]
  unsigned short* qMaxU = nullptr;     //[nUnattachedClusters]
  unsigned char* flagsU = nullptr;     //[nUnattachedClusters]
  unsigned short* padDiffU = nullptr;  //[nUnattachedClusters]
  unsigned int* timeDiffU = nullptr;   //[nUnattachedClusters]
  unsigned char* sigmaPadU = nullptr;  //[nUnattachedClusters]
  unsigned char* sigmaTimeU = nullptr; //[nUnattachedClusters]

  unsigned short* nTrackClusters = nullptr;  //[nTracks]
  unsigned int* nSliceRowClusters = nullptr; //[nSliceRows]

  ClassDefNV(CompressedClustersPtrs_helper, 1);
};

struct CompressedClustersDummy_helper {
};

//Version with valid ROOT streamers for storage
using CompressedClusters = CompressedClustersPtrs_helper<CompressedClustersCounters>;
//Slightly smaller version with pointers only for GPU constant cache
using CompressedClustersPtrsOnly = CompressedClustersPtrs_helper<CompressedClustersDummy_helper>;
} // namespace TPC
} // namespace o2

#endif
