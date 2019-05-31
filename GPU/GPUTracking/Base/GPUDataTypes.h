// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDataTypes.h
/// \author David Rohr

#ifndef GPUDATATYPES_H
#define GPUDATATYPES_H

#include "GPUCommonDef.h"

#if (!defined(__OPENCL__) || defined(__OPENCLCPP__)) && (!(defined(__CINT__) || defined(__ROOTCINT__)) || defined(__CLING__))
#define GPUDATATYPES_NOCOMPAT
#include "GPUTRDDef.h"

class AliHLTTPCClusterMCLabel;
struct AliHLTTPCRawCluster;
namespace o2
{
namespace tpc
{
struct ClusterNativeAccessFullTPC;
template <class T>
struct CompressedClustersPtrs_helper;
struct CompressedClustersCounters;
using CompressedClusters = CompressedClustersPtrs_helper<CompressedClustersCounters>;
} // namespace tpc
} // namespace o2
#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{
#ifdef GPUDATATYPES_NOCOMPAT
#include "utils/bitfield.h"
#define ENUM_CLASS class
#define ENUM_UINT : unsigned int
#define GPUCA_RECO_STEP GPUDataTypes::RecoStep
#else
#define ENUM_CLASS
#define ENUM_UINT
#define GPUCA_RECO_STEP GPUDataTypes
#endif

class GPUTPCSliceOutput;
class GPUTPCSliceOutTrack;
class GPUTPCSliceOutCluster;
class GPUTPCGMMergedTrack;
struct GPUTPCGMMergedTrackHit;
class GPUTRDTrackletWord;
class GPUTPCMCInfo;
struct GPUTPCClusterData;
struct ClusterNativeAccessExt;
struct GPUTRDTrackletLabels;

class GPUDataTypes
{
 public:
  enum ENUM_CLASS GeometryType ENUM_UINT{ RESERVED_GEOMETRY = 0, ALIROOT = 1, O2 = 2 };
  enum DeviceType ENUM_UINT { INVALID_DEVICE = 0,
                              CPU = 1,
                              CUDA = 2,
                              HIP = 3,
                              OCL = 4 };
  enum ENUM_CLASS RecoStep { TPCConversion = 1,
                             TPCSliceTracking = 2,
                             TPCMerging = 4,
                             TPCCompression = 8,
                             TRDTracking = 16,
                             ITSTracking = 32,
                             TPCdEdx = 64,
                             AllRecoSteps = 0x7FFFFFFF,
                             NoRecoStep = 0 };
  enum ENUM_CLASS InOutType { TPCClusters = 1,
                              TPCSectorTracks = 2,
                              TPCMergedTracks = 4,
                              TPCCompressedClusters = 8,
                              TRDTracklets = 16,
                              TRDTracks = 32 };

#ifdef GPUDATATYPES_NOCOMPAT
  static constexpr const char* const RECO_STEP_NAMES[] = { "TPC Transformation", "TPC Sector Tracking", "TPC Track Merging and Fit", "TPC Compression", "TRD Tracking", "ITS Tracking", "TPC dEdx Computation" };
  typedef bitfield<RecoStep, unsigned int> RecoStepField;
  typedef bitfield<InOutType, unsigned int> InOutTypeField;
#endif

  static DeviceType GetDeviceType(const char* type);
};

#ifdef GPUDATATYPES_NOCOMPAT
struct GPURecoStepConfiguration {
  GPUDataTypes::RecoStepField steps = 0;
  GPUDataTypes::InOutTypeField inputs = 0;
  GPUDataTypes::InOutTypeField outputs = 0;
};

struct GPUTrackingInOutPointers {
  GPUTrackingInOutPointers() = default;
  GPUTrackingInOutPointers(const GPUTrackingInOutPointers&) = default;
  static constexpr unsigned int NSLICES = 36;

  const GPUTPCClusterData* clusterData[NSLICES] = { nullptr };
  unsigned int nClusterData[NSLICES] = { 0 };
  const AliHLTTPCRawCluster* rawClusters[NSLICES] = { nullptr };
  unsigned int nRawClusters[NSLICES] = { 0 };
  const o2::tpc::ClusterNativeAccessFullTPC* clustersNative = nullptr;
  const GPUTPCSliceOutTrack* sliceOutTracks[NSLICES] = { nullptr };
  unsigned int nSliceOutTracks[NSLICES] = { 0 };
  const GPUTPCSliceOutCluster* sliceOutClusters[NSLICES] = { nullptr };
  unsigned int nSliceOutClusters[NSLICES] = { 0 };
  const AliHLTTPCClusterMCLabel* mcLabelsTPC = nullptr;
  unsigned int nMCLabelsTPC = 0;
  const GPUTPCMCInfo* mcInfosTPC = nullptr;
  unsigned int nMCInfosTPC = 0;
  const GPUTPCGMMergedTrack* mergedTracks = nullptr;
  unsigned int nMergedTracks = 0;
  const GPUTPCGMMergedTrackHit* mergedTrackHits = nullptr;
  unsigned int nMergedTrackHits = 0;
  const GPUTRDTrack* trdTracks = nullptr;
  const o2::tpc::CompressedClusters* tpcCompressedClusters = nullptr;
  unsigned int nTRDTracks = 0;
  const GPUTRDTrackletWord* trdTracklets = nullptr;
  unsigned int nTRDTracklets = 0;
  const GPUTRDTrackletLabels* trdTrackletsMC = nullptr;
  unsigned int nTRDTrackletsMC = 0;
  friend class GPUReconstruction;
};
#undef GPUDATATYPES_NOCOMPAT
#endif

#undef ENUM_CLASS
#undef ENUM_UINT
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
