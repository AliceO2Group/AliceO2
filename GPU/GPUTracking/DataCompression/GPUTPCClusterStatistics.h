// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

namespace o2
{
namespace tpc
{
struct ClusterNativeAccess;
}
} // namespace o2

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUTPCClusterStatistics
{
 public:
#ifndef HAVE_O2HEADERS
  void RunStatistics(const o2::tpc::ClusterNativeAccess* clustersNative, const o2::tpc::CompressedClusters* clustersCompressed, const GPUParam& param){};
  void Finish(){};
#else
  static constexpr unsigned int NSLICES = GPUCA_NSLICES;
  void RunStatistics(const o2::tpc::ClusterNativeAccess* clustersNative, const o2::tpc::CompressedClusters* clustersCompressed, const GPUParam& param);
  void Finish();

 protected:
  template <class T, int I = 0>
  void FillStatistic(std::vector<int>& p, const T* ptr, size_t n);
  template <class T, class S, int I = 0>
  void FillStatisticCombined(std::vector<int>& p, const T* ptr1, const S* ptr2, size_t n, int max1);
  float Analyze(std::vector<int>& p, const char* name, bool count = true);

  TPCClusterDecompressor mDecoder;
  bool mDecodingError = false;

  std::vector<int> mPqTotA = std::vector<int>(5 * 5 * 1024, 0);
  std::vector<int> mPqMaxA = std::vector<int>(1024, 0);
  std::vector<int> mPflagsA = std::vector<int>(1 << 8, 0);
  std::vector<int> mProwDiffA = std::vector<int>(GPUCA_ROW_COUNT, 0);
  std::vector<int> mPsliceLegDiffA = std::vector<int>(GPUCA_NSLICES * 2, 0);
  std::vector<int> mPpadResA = std::vector<int>(1 << 16, 0);
  std::vector<int> mPtimeResA = std::vector<int>(1 << 24, 0);
  std::vector<int> mPsigmaPadA = std::vector<int>(1 << 8, 0);
  std::vector<int> mPsigmaTimeA = std::vector<int>(1 << 8, 0);
  std::vector<int> mPqPtA = std::vector<int>(1 << 8, 0);
  std::vector<int> mProwA = std::vector<int>(GPUCA_ROW_COUNT, 0);
  std::vector<int> mPsliceA = std::vector<int>(GPUCA_NSLICES, 0);
  std::vector<int> mPtimeA = std::vector<int>(1 << 24, 0);
  std::vector<int> mPpadA = std::vector<int>(1 << 16, 0);
  std::vector<int> mPqTotU = std::vector<int>(5 * 5 * 1024, 0);
  std::vector<int> mPqMaxU = std::vector<int>(1024, 0);
  std::vector<int> mPflagsU = std::vector<int>(1 << 8, 0);
  std::vector<int> mPpadDiffU = std::vector<int>(1 << 16, 0);
  std::vector<int> mPtimeDiffU = std::vector<int>(1 << 24, 0);
  std::vector<int> mPsigmaPadU = std::vector<int>(1 << 8, 0);
  std::vector<int> mPsigmaTimeU = std::vector<int>(1 << 8, 0);
  std::vector<int> mPnTrackClusters;
  std::vector<int> mPnSliceRowClusters;
  std::vector<int> mPsigmaU = std::vector<int>(1 << 16, 0);
  std::vector<int> mPsigmaA = std::vector<int>(1 << 16, 0);
  std::vector<int> mProwSliceA = std::vector<int>(GPUCA_ROW_COUNT * GPUCA_NSLICES * 2, 0);

  double mEntropy = 0;
  double mHuffman = 0;
  size_t mNTotalClusters = 0;
#endif
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
