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
#include "TPCPadGainCalib.h"

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
struct GPUTPCClusterMCInterim;
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
  void SetNMaxDigits(size_t nDigits, size_t nPages, size_t nDigitsFragment);

  void PrepareMC();
  void clearMCMemory();
#endif
  unsigned char* mPzs = nullptr;
  ZSOffset* mPzsOffsets = nullptr;
  MinMaxCN* mMinMaxCN = nullptr;
  unsigned char* mPpadHasLostBaseline = nullptr;
  tpc::Digit* mPdigits = nullptr; // input digits, only set if ZS is skipped
  ChargePos* mPpositions = nullptr;
  ChargePos* mPpeakPositions = nullptr;
  ChargePos* mPfilteredPeakPositions = nullptr;
  unsigned char* mPisPeak = nullptr;
  uint* mPclusterPosInRow = nullptr; // store the index where the corresponding cluster is stored in a bucket.
                                     // Required when MC are enabled to write the mc data to the correct position.
                                     // Set to >= mNMaxClusterPerRow if cluster was discarded.
  ushort* mPchargeMap = nullptr;
  unsigned char* mPpeakMap = nullptr;
  uint* mPindexMap = nullptr;
  uint* mPclusterInRow = nullptr;
  tpc::ClusterNative* mPclusterByRow = nullptr;
  GPUTPCClusterMCInterim* mPlabelsByRow = nullptr;
  int* mPbuf = nullptr;
  Memory* mPmemory = nullptr;

  o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel> const* mPinputLabels = nullptr;
  uint* mPlabelsInRow = nullptr;
  uint mPlabelsHeaderGlobalOffset = 0;
  uint mPlabelsDataGlobalOffset = 0;

  int mISlice = 0;
  constexpr static int mScanWorkGroupSize = GPUCA_THREAD_COUNT_SCAN;
  unsigned int mNMaxClusterPerRow = 0;
  unsigned int mNMaxClusters = 0;
  size_t mNMaxPages = 0;
  size_t mNMaxDigits = 0;
  size_t mNMaxDigitsFragment = 0;
  size_t mNMaxPeaks = 0;
  size_t mBufSize = 0;
  unsigned int mNBufs = 0;

  short mMemoryId = -1;
  short mScratchId = -1;
  short mZSId = -1;
  short mZSOffsetId = -1;
  short mOutputId = -1;

  GPUdi() const GPUTPCGeometry* getGeometry() const;

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

} // namespace GPUCA_NAMESPACE::gpu

#endif
