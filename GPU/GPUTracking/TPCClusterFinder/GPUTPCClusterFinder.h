// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
} // namespace dataformats

namespace tpc
{
struct ClusterNative;
class Digit;
} // namespace tpc

} // namespace o2

namespace GPUCA_NAMESPACE
{
namespace gpu
{
struct GPUTPCClusterMCInterim;

struct ChargePos;

class GPUTPCClusterFinder : public GPUProcessor
{
 public:
  struct Memory {
    struct counters_t {
      size_t nDigits = 0;
      size_t nPositions = 0; // TODO use this instead of nDigits
      size_t nPeaks = 0;
      size_t nClusters = 0;
      unsigned int maxTimeBin = 0;
      unsigned int nPages = 0;
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
  void* SetPointersZSOffset(void* mem);

  size_t getNSteps(size_t items) const;
  void SetNMaxDigits(size_t nDigits, size_t nPages);

  void PrepareMC();
  void clearMCMemory();
#endif
  unsigned char* mPzs = nullptr;
  ZSOffset* mPzsOffsets = nullptr;
  MinMaxCN* mMinMaxCN = nullptr;
  tpc::Digit* mPdigits = nullptr; // input digits, only set if ZS is skipped
  ChargePos* mPpositions = nullptr;
  ChargePos* mPpeakPositions = nullptr;
  ChargePos* mPfilteredPeakPositions = nullptr;
  unsigned char* mPisPeak = nullptr;
  ushort* mPchargeMap = nullptr;
  unsigned char* mPpeakMap = nullptr;
  uint* mPindexMap = nullptr;
  uint* mPclusterInRow = nullptr;
  tpc::ClusterNative* mPclusterByRow = nullptr;
  GPUTPCClusterMCInterim* mPlabelsByRow = nullptr;
  int* mPbuf = nullptr;
  Memory* mPmemory = nullptr;

  o2::dataformats::MCTruthContainer<o2::MCCompLabel> const* mPinputLabels = nullptr;
  uint* mPlabelHeaderOffset = nullptr;
  uint* mPlabelDataOffset = nullptr;

  int mISlice = 0;
  constexpr static int mScanWorkGroupSize = GPUCA_THREAD_COUNT_SCAN;
  size_t mNMaxClusterPerRow = 0;
  size_t mNMaxPages = 0;
  size_t mNMaxDigits = 0;
  size_t mNMaxPeaks = 0;
  size_t mNMaxClusters = 0;
  size_t mBufSize = 0;
  size_t mNBufs = 0;

  short mMemoryId = -1;
  short mZSOffsetId = -1;
  short mOutputId = -1;

#ifndef GPUCA_GPUCODE
  void DumpDigits(std::ostream& out);
  void DumpChargeMap(std::ostream& out, std::string_view);
  void DumpPeaks(std::ostream& out);
  void DumpPeaksCompacted(std::ostream& out);
  void DumpSuppressedPeaks(std::ostream& out);
  void DumpSuppressedPeaksCompacted(std::ostream& out);
  void DumpCountedPeaks(std::ostream& out);
  void DumpClusters(std::ostream& out);
#endif
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
