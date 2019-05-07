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
struct CompressedClusters {
  unsigned int nTracks = 0;
  unsigned int nAttachedClusters = 0;
  unsigned int nUnattachedClusters = 0;

  unsigned short* qTotA = nullptr;
  unsigned short* qMaxA = nullptr;
  unsigned char* flagsA = nullptr;
  unsigned char* rowDiffA = nullptr;
  unsigned char* sliceLegDiffA = nullptr;
  unsigned short* padResA = nullptr;
  unsigned int* timeResA = nullptr;
  unsigned char* sigmaPadA = nullptr;
  unsigned char* sigmaTimeA = nullptr;

  char* qPtA = nullptr;
  unsigned char* rowA = nullptr;
  unsigned char* sliceA = nullptr;
  unsigned int* timeA = nullptr;
  unsigned short* padA = nullptr;

  unsigned short* qTotU = nullptr;
  unsigned short* qMaxU = nullptr;
  unsigned char* flagsU = nullptr;
  unsigned short* padDiffU = nullptr;
  unsigned int* timeDiffU = nullptr;
  unsigned char* sigmaPadU = nullptr;
  unsigned char* sigmaTimeU = nullptr;

  unsigned short* nTrackClusters = nullptr;
  unsigned int* nSliceRowClusters = nullptr;
};
} // namespace TPC
} // namespace o2

#endif
