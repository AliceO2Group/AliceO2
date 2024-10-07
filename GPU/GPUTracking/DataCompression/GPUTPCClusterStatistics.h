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

/// \file GPUTPCClusterStatistics.h
/// \author David Rohr

#ifndef GPUTPCCLUSTERSTATISTICS_H
#define GPUTPCCLUSTERSTATISTICS_H

#include "GPUTPCCompression.h"
#include "TPCClusterDecompressor.h"
#include <vector>

namespace o2::tpc
{
struct ClusterNativeAccess;
} // namespace o2::tpc

namespace GPUCA_NAMESPACE::gpu
{
class GPUTPCClusterStatistics
{
 public:
#ifndef GPUCA_HAVE_O2HEADERS
  void RunStatistics(const o2::tpc::ClusterNativeAccess* clustersNative, const o2::tpc::CompressedClusters* clustersCompressed, const GPUParam& param){};
  void Finish(){};
#else
  static constexpr uint32_t NSLICES = GPUCA_NSLICES;
  void RunStatistics(const o2::tpc::ClusterNativeAccess* clustersNative, const o2::tpc::CompressedClusters* clustersCompressed, const GPUParam& param);
  void Finish();

 protected:
  template <class T, int32_t I = 0>
  void FillStatistic(std::vector<int32_t>& p, const T* ptr, size_t n);
  template <class T, class S, int32_t I = 0>
  void FillStatisticCombined(std::vector<int32_t>& p, const T* ptr1, const S* ptr2, size_t n, int32_t max1);
  float Analyze(std::vector<int32_t>& p, const char* name, bool count = true);

  TPCClusterDecompressor mDecoder;
  bool mDecodingError = false;

  static constexpr uint32_t P_MAX_QMAX = GPUTPCCompression::P_MAX_QMAX;
  static constexpr uint32_t P_MAX_QTOT = GPUTPCCompression::P_MAX_QTOT;
  static constexpr uint32_t P_MAX_TIME = GPUTPCCompression::P_MAX_TIME;
  static constexpr uint32_t P_MAX_PAD = GPUTPCCompression::P_MAX_PAD;
  static constexpr uint32_t P_MAX_SIGMA = GPUTPCCompression::P_MAX_SIGMA;
  static constexpr uint32_t P_MAX_FLAGS = GPUTPCCompression::P_MAX_FLAGS;
  static constexpr uint32_t P_MAX_QPT = GPUTPCCompression::P_MAX_QPT;

  std::vector<int32_t> mPqTotA = std::vector<int32_t>(P_MAX_QTOT, 0);
  std::vector<int32_t> mPqMaxA = std::vector<int32_t>(P_MAX_QMAX, 0);
  std::vector<int32_t> mPflagsA = std::vector<int32_t>(P_MAX_FLAGS, 0);
  std::vector<int32_t> mProwDiffA = std::vector<int32_t>(GPUCA_ROW_COUNT, 0);
  std::vector<int32_t> mPsliceLegDiffA = std::vector<int32_t>(GPUCA_NSLICES * 2, 0);
  std::vector<int32_t> mPpadResA = std::vector<int32_t>(P_MAX_PAD, 0);
  std::vector<int32_t> mPtimeResA = std::vector<int32_t>(P_MAX_TIME, 0);
  std::vector<int32_t> mPsigmaPadA = std::vector<int32_t>(P_MAX_SIGMA, 0);
  std::vector<int32_t> mPsigmaTimeA = std::vector<int32_t>(P_MAX_SIGMA, 0);
  std::vector<int32_t> mPqPtA = std::vector<int32_t>(P_MAX_QPT, 0);
  std::vector<int32_t> mProwA = std::vector<int32_t>(GPUCA_ROW_COUNT, 0);
  std::vector<int32_t> mPsliceA = std::vector<int32_t>(GPUCA_NSLICES, 0);
  std::vector<int32_t> mPtimeA = std::vector<int32_t>(P_MAX_TIME, 0);
  std::vector<int32_t> mPpadA = std::vector<int32_t>(P_MAX_PAD, 0);
  std::vector<int32_t> mPqTotU = std::vector<int32_t>(P_MAX_QTOT, 0);
  std::vector<int32_t> mPqMaxU = std::vector<int32_t>(P_MAX_QMAX, 0);
  std::vector<int32_t> mPflagsU = std::vector<int32_t>(P_MAX_FLAGS, 0);
  std::vector<int32_t> mPpadDiffU = std::vector<int32_t>(P_MAX_PAD, 0);
  std::vector<int32_t> mPtimeDiffU = std::vector<int32_t>(P_MAX_TIME, 0);
  std::vector<int32_t> mPsigmaPadU = std::vector<int32_t>(P_MAX_SIGMA, 0);
  std::vector<int32_t> mPsigmaTimeU = std::vector<int32_t>(P_MAX_SIGMA, 0);
  std::vector<int32_t> mPnTrackClusters;
  std::vector<int32_t> mPnSliceRowClusters;
  std::vector<int32_t> mPsigmaU = std::vector<int32_t>(P_MAX_SIGMA * P_MAX_SIGMA, 0);
  std::vector<int32_t> mPsigmaA = std::vector<int32_t>(P_MAX_SIGMA * P_MAX_SIGMA, 0);
  std::vector<int32_t> mPQU = std::vector<int32_t>(P_MAX_QMAX * P_MAX_QTOT, 0);
  std::vector<int32_t> mPQA = std::vector<int32_t>(P_MAX_QMAX * P_MAX_QTOT, 0);
  std::vector<int32_t> mProwSliceA = std::vector<int32_t>(GPUCA_ROW_COUNT * GPUCA_NSLICES * 2, 0);

  double mEntropy = 0;
  double mHuffman = 0;
  size_t mNTotalClusters = 0;
#endif
};
} // namespace GPUCA_NAMESPACE::gpu

#endif
