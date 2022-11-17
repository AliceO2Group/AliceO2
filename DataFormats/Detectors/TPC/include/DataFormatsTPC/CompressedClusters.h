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

/// \file CompressedClusters.h
/// \brief Container to store compressed TPC cluster data
/// \author David Rohr

#ifndef ALICEO2_DATAFORMATSTPC_COMPRESSED_CLUSTERS_H
#define ALICEO2_DATAFORMATSTPC_COMPRESSED_CLUSTERS_H

#include "GPUCommonDef.h"
#include "GPUCommonRtypes.h"

namespace o2
{
namespace tpc
{
struct CompressedClustersCounters {
  unsigned int nTracks = 0;
  unsigned int nAttachedClusters = 0;
  unsigned int nUnattachedClusters = 0;
  unsigned int nAttachedClustersReduced = 0;
  unsigned int nSliceRows = 36 * 152;
  unsigned char nComppressionModes = 0;
  float solenoidBz = -1e6f;
  int maxTimeBin = -1e6;

  ClassDefNV(CompressedClustersCounters, 3);
};

template <class TCHAR, class TSHORT, class TINT>
struct CompressedClustersPtrs_x {
  TSHORT qTotA = 0;        //!
  TSHORT qMaxA = 0;        //!
  TCHAR flagsA = 0;        //!
  TCHAR rowDiffA = 0;      //!
  TCHAR sliceLegDiffA = 0; //!
  TSHORT padResA = 0;      //!
  TINT timeResA = 0;       //!
  TCHAR sigmaPadA = 0;     //!
  TCHAR sigmaTimeA = 0;    //!

  TCHAR qPtA = 0;   //!
  TCHAR rowA = 0;   //!
  TCHAR sliceA = 0; //!
  TINT timeA = 0;   //!
  TSHORT padA = 0;  //!

  TSHORT qTotU = 0;     //!
  TSHORT qMaxU = 0;     //!
  TCHAR flagsU = 0;     //!
  TSHORT padDiffU = 0;  //!
  TINT timeDiffU = 0;   //!
  TCHAR sigmaPadU = 0;  //!
  TCHAR sigmaTimeU = 0; //!

  TSHORT nTrackClusters = 0;  //!
  TINT nSliceRowClusters = 0; //!

  ClassDefNV(CompressedClustersPtrs_x, 3);
};

struct CompressedClustersPtrs : public CompressedClustersPtrs_x<unsigned char*, unsigned short*, unsigned int*> {
};

struct CompressedClustersOffsets : public CompressedClustersPtrs_x<size_t, size_t, size_t> {
};

struct CompressedClustersFlat;

struct CompressedClusters : public CompressedClustersCounters, public CompressedClustersPtrs { // TODO: Need a const version of this, currently the constructor allows to create a non-const version from const CompressedClustersFlat, which should not be allowed
  CompressedClusters() CON_DEFAULT;
  ~CompressedClusters() CON_DEFAULT;
  CompressedClusters(const CompressedClustersFlat& c);

  void dump();

  ClassDefNV(CompressedClusters, 3);
};

struct CompressedClustersROOT : public CompressedClusters {
  CompressedClustersROOT() CON_DEFAULT;
  CompressedClustersROOT(const CompressedClustersFlat& v) : CompressedClusters(v) {}
  CompressedClustersROOT(const CompressedClusters& v) : CompressedClusters(v) {}
  // flatbuffer used for streaming
  int flatdataSize = 0;
  char* flatdata = nullptr; //[flatdataSize]

  ClassDefNV(CompressedClustersROOT, 3);
};

struct CompressedClustersFlat : private CompressedClustersCounters, private CompressedClustersOffsets {
  friend struct CompressedClusters;               // We don't want anyone to access the members directly, should only be used to construct a CompressedClusters struct
  CompressedClustersFlat() CON_DELETE;            // Must not be constructed
  size_t totalDataSize = 0;                       // Total data size of header + content
  const CompressedClusters* ptrForward = nullptr; // Must be 0 if this object is really flat, or can be a ptr to a CompressedClusters struct (abusing the flat structure to forward a ptr to the e.g. root version)

  void set(size_t bufferSize, const CompressedClusters& v);
  void setForward(const CompressedClusters* p);
};

} // namespace tpc
} // namespace o2

#endif
