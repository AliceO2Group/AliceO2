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

/// \file GPUTPCClusterFinder.h
/// \author David Rohr

#ifndef O2_GPU_GPUTPCCLUSTERFINDER_H
#define O2_GPU_GPUTPCCLUSTERFINDER_H

#include "GPUDef.h"
#include "GPUProcessor.h"
#include "GPUDataTypes.h"
#include "CfFragment.h"

namespace o2
{

class MCCompLabel;

namespace dataformats
{
template <typename TruthElement>
class MCTruthContainer;
template <typename TruthElement>
class ConstMCTruthContainerView;
} // namespace dataformats

namespace tpc
{
struct ClusterNative;
class Digit;
} // namespace tpc

} // namespace o2

namespace GPUCA_NAMESPACE::gpu
{
struct GPUTPCClusterMCInterimArray;
struct TPCPadGainCalib;

struct ChargePos;

class GPUTPCGeometry;

class GPUTPCClusterFinder : public GPUProcessor
{
 public:
  struct Memory {
    struct counters_t {
      size_t nDigits = 0;
      tpccf::SizeT nDigitsInFragment = 0; // num of digits in fragment can differ from nPositions if ZS is active
      tpccf::SizeT nPositions = 0;
      tpccf::SizeT nPeaks = 0;
      tpccf::SizeT nClusters = 0;
      unsigned int maxTimeBin = 0;
      unsigned int nPagesSubslice = 0;
    } counters;
    CfFragment fragment;
  };

  struct ZSOffset {
    unsigned int offset;
    unsigned short endpoint;
    unsigned short num;
  };

  struct MinMaxCN {
    unsigned int minC, minN, maxC, maxN;
  };

#ifndef GPUCA_GPUCODE
  ~GPUTPCClusterFinder();
  void InitializeProcessor();
  void RegisterMemoryAllocation();
  void SetMaxData(const GPUTrackingInOutPointers& io);

  void* SetPointersInput(void* mem);
  void* SetPointersOutput(void* mem);
  void* SetPointersScratch(void* mem);
  void* SetPointersMemory(void* mem);
  void* SetPointersZS(void* mem);
  void* SetPointersZSOffset(void* mem);

  unsigned int getNSteps(size_t items) const;
  void SetNMaxDigits(size_t nDigits, size_t nPages, size_t nDigitsFragment, size_t nDigitsEndpointMax);

  void PrepareMC();
  void clearMCMemory();
#endif
  unsigned char* mPzs = nullptr;
  ZSOffset* mPzsOffsets = nullptr;
  MinMaxCN* mMinMaxCN = nullptr;
  unsigned char* mPpadIsNoisy = nullptr;
  tpc::Digit* mPdigits = nullptr; // input digits, only set if ZS is skipped
  ChargePos* mPpositions = nullptr;
  ChargePos* mPpeakPositions = nullptr;
  ChargePos* mPfilteredPeakPositions = nullptr;
  unsigned char* mPisPeak = nullptr;
  unsigned int* mPclusterPosInRow = nullptr; // store the index where the corresponding cluster is stored in a bucket.
                                             // Required when MC are enabled to write the mc data to the correct position.
                                             // Set to >= mNMaxClusterPerRow if cluster was discarded.
  unsigned short* mPchargeMap = nullptr;
  unsigned char* mPpeakMap = nullptr;
  unsigned int* mPindexMap = nullptr;
  unsigned int* mPclusterInRow = nullptr;
  tpc::ClusterNative* mPclusterByRow = nullptr;
  GPUTPCClusterMCInterimArray* mPlabelsByRow = nullptr;
  int* mPbuf = nullptr;
  Memory* mPmemory = nullptr;

  o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel> const* mPinputLabels = nullptr;
  unsigned int* mPlabelsInRow = nullptr;
  unsigned int mPlabelsHeaderGlobalOffset = 0;
  unsigned int mPlabelsDataGlobalOffset = 0;

  int mISlice = 0;
  constexpr static int mScanWorkGroupSize = GPUCA_THREAD_COUNT_SCAN;
  unsigned int mNMaxClusterPerRow = 0;
  unsigned int mNMaxClusters = 0;
  unsigned int mNMaxPages = 0;
  size_t mNMaxDigits = 0;
  size_t mNMaxDigitsFragment = 0;
  size_t mNMaxDigitsEndpoint = 0;
  size_t mNMaxPeaks = 0;
  size_t mBufSize = 0;
  unsigned int mNBufs = 0;

  short mMemoryId = -1;
  short mScratchId = -1;
  short mZSId = -1;
  short mZSOffsetId = -1;
  short mOutputId = -1;

#ifndef GPUCA_GPUCODE
  void DumpDigits(std::ostream& out);
  void DumpChargeMap(std::ostream& out, std::string_view, bool doGPU);
  void DumpPeaks(std::ostream& out);
  void DumpPeaksCompacted(std::ostream& out);
  void DumpSuppressedPeaks(std::ostream& out);
  void DumpSuppressedPeaksCompacted(std::ostream& out);
  void DumpCountedPeaks(std::ostream& out);
  void DumpClusters(std::ostream& out);
#endif
};

} // namespace GPUCA_NAMESPACE::gpu

#endif
