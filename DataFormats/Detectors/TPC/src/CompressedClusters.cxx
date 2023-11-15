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

/// \file CompressedClusters.cxx
/// \brief Container to store compressed TPC cluster data

#include "DataFormatsTPC/CompressedClusters.h"
#include "GPUCommonLogger.h"
#include <cstring>
#ifndef GPUCA_STANDALONE
#include "DataFormatsTPC/CompressedClustersHelpers.h"
#include "TBuffer.h"
#include <vector>
#include <gsl/span>
#endif

namespace o2::tpc
{

CompressedClusters::CompressedClusters(const CompressedClustersFlat& c)
{
  CompressedClustersCounters& cc = *this;
  CompressedClustersPtrs& cp = *this;
  const CompressedClustersOffsets& offsets = c;
  cc = c;
  for (unsigned int i = 0; i < sizeof(cp) / sizeof(size_t); i++) {
    reinterpret_cast<void**>(&cp)[i] = reinterpret_cast<void*>(reinterpret_cast<const size_t*>(&offsets)[i] + reinterpret_cast<size_t>(&c)); // Restore pointers from offsets
  }
};

void CompressedClustersFlat::set(size_t bufferSize, const CompressedClusters& v)
{
  CompressedClustersCounters& cc = *this;
  CompressedClustersOffsets& offsets = *this;
  const CompressedClustersPtrs& ptrs = v;
  cc = v;
  for (unsigned int i = 0; i < sizeof(offsets) / sizeof(size_t); i++) {
    reinterpret_cast<size_t*>(&offsets)[i] = (reinterpret_cast<const size_t*>(&ptrs)[i] - reinterpret_cast<size_t>(this)); // Compute offsets from beginning of structure
  }
  ptrForward = nullptr;
  totalDataSize = bufferSize;
}

void CompressedClustersFlat::setForward(const CompressedClusters* p)
{
  memset((void*)this, 0, sizeof(*this));
  ptrForward = p;
}

void CompressedClusters::dump()
{
  LOG(info) << "nTracks" << nTracks;
  for (unsigned int i = 0; i < nTracks; i++) {
    LOG(info) << "  " << i << ":" << (unsigned int)qPtA[i] << " " << (unsigned int)rowA[i] << " " << (unsigned int)sliceA[i] << " " << (unsigned int)timeA[i] << " " << (unsigned int)padA[i];
  }
  LOG(info) << "nAttachedClusters" << nAttachedClusters;
  for (unsigned int i = 0; i < nAttachedClusters; i++) {
    LOG(info) << "  " << i << ":" << (unsigned int)qTotA[i] << " " << (unsigned int)qMaxA[i] << " " << (unsigned int)flagsA[i] << " " << (unsigned int)sigmaPadA[i] << " " << (unsigned int)sigmaTimeA[i];
  }
  LOG(info) << "nAttachedClustersReduced" << nAttachedClustersReduced;
  for (unsigned int i = 0; i < nAttachedClustersReduced; i++) {
    LOG(info) << "  " << i << ":" << (unsigned int)rowDiffA[i] << " " << (unsigned int)sliceLegDiffA[i] << " " << (unsigned int)padResA[i] << " " << (unsigned int)timeResA[i];
  }
  LOG(info) << "nUnattachedClusters" << nUnattachedClusters;
  for (unsigned int i = 0; i < nUnattachedClusters; i++) {
    LOG(info) << "  " << i << ":" << (unsigned int)qTotU[i] << " " << (unsigned int)qMaxU[i] << " " << (unsigned int)flagsU[i] << " " << (unsigned int)padDiffU[i] << " " << (unsigned int)timeDiffU[i] << " " << (unsigned int)sigmaPadU[i] << " " << (unsigned int)sigmaTimeU[i];
  }
}

#ifndef GPUCA_STANDALONE
void CompressedClustersROOT::Streamer(TBuffer& R__b)
{
  // the custom streamer for CompressedClustersROOT
  if (R__b.IsReading()) {
    R__b.ReadClassBuffer(CompressedClustersROOT::Class(), this);
    gsl::span flatdata{this->flatdata, static_cast<std::size_t>(this->flatdataSize)};
    CompressedClustersHelpers::restoreFrom(flatdata, *this);
  } else {
    std::vector<char> flatdata;
    // member flatdata is nullptr unless it is an already extracted object which
    // is streamed again. Overriding the existing flatbuffer is a potential
    // memory leak
    // Note: lots of cornercases here which are diffucult to catch with the design
    // of CompClusters container (which is optimized to be used with GPU). Main concern
    // is if an extracted object is not read-only and the counters are changed.
    // TODO: we can add a consistency check whether the size of the flatbuffer
    // and the pointers match the object.
    bool isflat = this->flatdata != nullptr;
    if (!isflat) {
      CompressedClustersHelpers::flattenTo(flatdata, *this);
      this->flatdataSize = flatdata.size();
      this->flatdata = flatdata.data();
    }
    R__b.WriteClassBuffer(CompressedClustersROOT::Class(), this);
    if (!isflat) {
      this->flatdataSize = 0;
      this->flatdata = nullptr;
    }
  }
}
#endif

} // namespace o2::tpc
