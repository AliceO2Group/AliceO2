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

#ifndef __OPENCL__
#include <cstddef>
#endif
#ifdef GPUCA_NOCOMPAT_ALLOPENCL
#include <type_traits>
#endif
#ifdef GPUCA_NOCOMPAT
#include "GPUTRDDef.h"

class AliHLTTPCClusterMCLabel;
struct AliHLTTPCRawCluster;
namespace o2
{
namespace tpc
{
struct ClusterNativeAccess;
template <class T>
struct CompressedClustersPtrs_helper;
struct CompressedClustersCounters;
using CompressedClusters = CompressedClustersPtrs_helper<CompressedClustersCounters>;
} // namespace tpc
} // namespace o2
#endif

namespace o2
{
class MCCompLabel;
namespace base
{
class MatLayerCylSet;
} // namespace base
namespace trd
{
class TRDGeometryFlat;
} // namespace trd
namespace dataformats
{
template <class T>
class MCTruthContainer;
} // namespace dataformats
} // namespace o2

namespace gpucf // TODO: Clean up namespace
{
typedef struct PackedDigit_s PackedDigit;
}

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class TPCFastTransform;
} // namespace gpu
} // namespace GPUCA_NAMESPACE

namespace GPUCA_NAMESPACE
{
namespace gpu
{
#ifdef GPUCA_NOCOMPAT_ALLOPENCL
#include "utils/bitfield.h"
#define ENUM_CLASS class
#define ENUM_UINT : unsigned int
#define GPUCA_RECO_STEP GPUDataTypes::RecoStep
#else
#define ENUM_CLASS
#define ENUM_UINT
#define GPUCA_RECO_STEP GPUDataTypes
#endif

#ifdef __OPENCL__
MEM_CLASS_PRE() // Macro with some template magic for OpenCL 1.2
#endif
class GPUTPCTrack;
class GPUTPCHitId;
class GPUTPCGMMergedTrack;
struct GPUTPCGMMergedTrackHit;
class GPUTRDTrackletWord;
class GPUTPCMCInfo;
struct GPUTPCClusterData;
struct GPUTRDTrackletLabels;

class GPUDataTypes
{
 public:
  enum ENUM_CLASS GeometryType ENUM_UINT{RESERVED_GEOMETRY = 0, ALIROOT = 1, O2 = 2};
  enum DeviceType ENUM_UINT { INVALID_DEVICE = 0,
                              CPU = 1,
                              CUDA = 2,
                              HIP = 3,
                              OCL = 4,
                              OCL2 = 5 };
  enum ENUM_CLASS RecoStep { TPCConversion = 1,
                             TPCSliceTracking = 2,
                             TPCMerging = 4,
                             TPCCompression = 8,
                             TRDTracking = 16,
                             ITSTracking = 32,
                             TPCdEdx = 64,
                             TPCClusterFinding = 128,
                             AllRecoSteps = 0x7FFFFFFF,
                             NoRecoStep = 0 };
  enum ENUM_CLASS InOutType { TPCClusters = 1,
                              TPCSectorTracks = 2,
                              TPCMergedTracks = 4,
                              TPCCompressedClusters = 8,
                              TRDTracklets = 16,
                              TRDTracks = 32,
                              TPCRaw = 64 };

#ifdef GPUCA_NOCOMPAT_ALLOPENCL
  static constexpr const char* const RECO_STEP_NAMES[] = {"TPC Transformation", "TPC Sector Tracking", "TPC Track Merging and Fit", "TPC Compression", "TRD Tracking", "ITS Tracking", "TPC dEdx Computation"};
  typedef bitfield<RecoStep, unsigned int> RecoStepField;
  typedef bitfield<InOutType, unsigned int> InOutTypeField;
#endif

  static DeviceType GetDeviceType(const char* type);
};

#ifdef GPUCA_NOCOMPAT_ALLOPENCL
struct GPURecoStepConfiguration {
  GPUDataTypes::RecoStepField steps = 0;
  GPUDataTypes::RecoStepField stepsGPUMask = GPUDataTypes::RecoStep::AllRecoSteps;
  GPUDataTypes::InOutTypeField inputs = 0;
  GPUDataTypes::InOutTypeField outputs = 0;
};
#endif

#ifdef GPUCA_NOCOMPAT
struct GPUCalibObjects {
  TPCFastTransform* fastTransform = nullptr;
  o2::base::MatLayerCylSet* matLUT = nullptr;
  o2::trd::TRDGeometryFlat* trdGeometry = nullptr;
};

struct GPUCalibObjectsConst { // TODO: Any chance to do this as template?
  const TPCFastTransform* fastTransform = nullptr;
  const o2::base::MatLayerCylSet* matLUT = nullptr;
  const o2::trd::TRDGeometryFlat* trdGeometry = nullptr;
};

struct GPUTrackingInOutPointers {
  GPUTrackingInOutPointers() = default;
  GPUTrackingInOutPointers(const GPUTrackingInOutPointers&) = default;
  static constexpr unsigned int NSLICES = 36;

  size_t tpcRaw = 0;
  const gpucf::PackedDigit* tpcDigits[NSLICES] = {nullptr};
  size_t nTPCDigits[NSLICES] = {0};
  const GPUTPCClusterData* clusterData[NSLICES] = {nullptr};
  unsigned int nClusterData[NSLICES] = {0};
  const AliHLTTPCRawCluster* rawClusters[NSLICES] = {nullptr};
  unsigned int nRawClusters[NSLICES] = {0};
  const o2::tpc::ClusterNativeAccess* clustersNative = nullptr;
  const GPUTPCTrack* sliceOutTracks[NSLICES] = {nullptr};
  unsigned int nSliceOutTracks[NSLICES] = {0};
  const GPUTPCHitId* sliceOutClusters[NSLICES] = {nullptr};
  unsigned int nSliceOutClusters[NSLICES] = {0};
  const AliHLTTPCClusterMCLabel* mcLabelsTPC = nullptr;
  unsigned int nMCLabelsTPC = 0;
  const GPUTPCMCInfo* mcInfosTPC = nullptr;
  unsigned int nMCInfosTPC = 0;
  const GPUTPCGMMergedTrack* mergedTracks = nullptr;
  unsigned int nMergedTracks = 0;
  const GPUTPCGMMergedTrackHit* mergedTrackHits = nullptr;
  unsigned int nMergedTrackHits = 0;
  const o2::tpc::CompressedClusters* tpcCompressedClusters = nullptr;
  const GPUTRDTrackletWord* trdTracklets = nullptr;
  unsigned int nTRDTracklets = 0;
  const GPUTRDTrackletLabels* trdTrackletsMC = nullptr;
  unsigned int nTRDTrackletsMC = 0;
  const GPUTRDTrack* trdTracks = nullptr;
  unsigned int nTRDTracks = 0;
};
#else
struct GPUTrackingInOutPointers {
};
struct GPUCalibObjectsConst {
};
#endif

#undef ENUM_CLASS
#undef ENUM_UINT
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
