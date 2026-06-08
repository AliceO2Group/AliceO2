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
#include "GPUDataTypesIO.h"
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
struct ClusterNativeAccess;
class Digit;
} // namespace tpc

} // namespace o2

namespace o2::gpu
{
struct GPUTPCClusterMCInterimArray;
struct TPCPadGainCalib;
struct HIPTailDescriptor;

struct CfChargePos;

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
      uint32_t maxTimeBin = 0;
      uint32_t nPagesSubsector = 0;
    } counters;
    CfFragment fragment;
  };

  struct ZSOffset {
    uint32_t offset;
    uint16_t endpoint;
    uint16_t num;
  };

  struct MinMaxCN {
    uint32_t zsPtrFirst, zsPageFirst, zsPtrLast, zsPageLast;
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

  uint32_t getNSteps(size_t items) const;
  void SetNMaxDigits(size_t nDigits, size_t nPages, size_t nDigitsFragment, size_t nDigitsEndpointMax);

  void AllocMCBuffers();
  void InitMCBuffersForFragment();
  void FreeMCBuffers();
#endif
  uint8_t* mPzs = nullptr;
  ZSOffset* mPzsOffsets = nullptr;
  MinMaxCN* mMinMaxCN = nullptr;
  uint8_t* mPpadIsNoisy = nullptr;
  tpc::Digit* mPdigits = nullptr; // input digits, only set if ZS is skipped
  CfChargePos* mPpositions = nullptr;
  CfChargePos* mPpeakPositions = nullptr;
  CfChargePos* mPfilteredPeakPositions = nullptr;
  uint8_t* mPisPeak = nullptr;
  uint32_t* mPclusterPosInRow = nullptr; // store the index where the corresponding cluster is stored in a bucket.
                                         // Required when MC are enabled to write the mc data to the correct position.
                                         // Set to >= mNMaxClusterPerRow if cluster was discarded.
  uint32_t* mPhipClusterPosInRow = nullptr; // Identical to mPclusterPosInRow. Need a seperate array for HIP cluster because tail index is used to identify clusters across GPU and CPU
  uint16_t* mPchargeMap = nullptr;
  uint8_t* mPpeakMap = nullptr;
  uint32_t* mPindexMap = nullptr;
  uint32_t* mPclusterInRow = nullptr;
  tpc::ClusterNative* mPclusterByRow = nullptr;
  GPUTPCClusterMCInterimArray* mPlabelsByRow = nullptr;
  int32_t* mPscanBuf = nullptr;
  HIPTailDescriptor* mPhipTailsByRow = nullptr;
  uint32_t* mPnHIPTails = nullptr; // one counter per row
  Memory* mPmemory = nullptr;

  GPUdi() int32_t* GetScanBuffer(int32_t iBuf) const { return mPscanBuf + iBuf * mBufSize; }

  o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel> const* mPinputLabels = nullptr;
  uint32_t* mPlabelsInRow = nullptr;
  uint32_t mPlabelsHeaderGlobalOffset = 0;
  uint32_t mPlabelsDataGlobalOffset = 0;

  uint32_t mISector = 0;
  uint32_t mNMaxClusterPerRow = 0;
  uint32_t mNMaxClusters = 0;
  uint32_t mNMaxPages = 0;
  size_t mNMaxDigits = 0;
  size_t mNMaxDigitsFragment = 0;
  size_t mNMaxDigitsEndpoint = 0;
  size_t mNMaxPeaks = 0;
  size_t mBufSize = 0;
  uint32_t mNBufs = 0;

  int16_t mMemoryId = -1;
  int16_t mScratchId = -1;
  int16_t mZSId = -1;
  int16_t mZSOffsetId = -1;
  int16_t mOutputId = -1;

#ifndef GPUCA_GPUCODE
  void DumpDigits(std::ostream& out);
  void DumpChargeMap(std::ostream& out, std::string_view);
  void DumpPeakMap(std::ostream& out, std::string_view);
  void DumpPeaks(std::ostream& out);
  void DumpPeaksCompacted(std::ostream& out);
  void DumpSuppressedPeaks(std::ostream& out);
  void DumpSuppressedPeaksCompacted(std::ostream& out);
  void DumpClusters(std::ostream& out);
#endif
};

} // namespace o2::gpu

#endif
