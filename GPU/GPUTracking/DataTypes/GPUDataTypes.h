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

// These are basic and non-comprex data types, which will also be visible on the GPU.
// Please add complex data types required on the host but not GPU to GPUHostDataTypes.h and forward-declare!
#ifndef GPUCA_GPUCODE_DEVICE
#include <cstddef>
#ifdef GPUCA_NOCOMPAT_ALLOPENCL
#include <type_traits>
#endif
#endif
#ifdef GPUCA_NOCOMPAT
#include "GPUTRDDef.h"

struct AliHLTTPCClusterMCLabel;
struct AliHLTTPCRawCluster;
namespace o2
{
namespace tpc
{
struct ClusterNativeAccess;
struct CompressedClustersFlat;
class Digit;
class TrackTPC;
namespace constants
{
} // namespace constants
} // namespace tpc
} // namespace o2
#endif

namespace o2
{
class MCCompLabel;
namespace base
{
class Propagator;
class MatLayerCylSet;
} // namespace base
namespace trd
{
class GeometryFlat;
} // namespace trd
namespace dataformats
{
template <class T>
class MCTruthContainer;
template <class T>
class ConstMCTruthContainerView;
} // namespace dataformats
} // namespace o2

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class TPCFastTransform;
class TPCdEdxCalibrationSplines;
struct TPCPadGainCalib;
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

#if defined(__OPENCL__) && !defined(__OPENCLCPP__)
MEM_CLASS_PRE() // Macro with some template magic for OpenCL 1.2
#endif
class GPUTPCTrack;
class GPUTPCHitId;
class GPUTPCGMMergedTrack;
struct GPUTPCGMMergedTrackHit;
struct GPUTPCGMMergedTrackHitXYZ;
class GPUTRDTrackletWord;
struct GPUTPCMCInfo;
struct GPUTPCClusterData;
struct GPUTRDTrackletLabels;
struct GPUTPCDigitsMCInput;

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
  enum ENUM_CLASS GeneralStep { Prepare = 1,
                                QA = 2 };

  enum ENUM_CLASS RecoStep { TPCConversion = 1,
                             TPCSliceTracking = 2,
                             TPCMerging = 4,
                             TPCCompression = 8,
                             TRDTracking = 16,
                             ITSTracking = 32,
                             TPCdEdx = 64,
                             TPCClusterFinding = 128,
                             TPCDecompression = 256,
                             Refit = 512,
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
  static constexpr const char* const RECO_STEP_NAMES[] = {"TPC Transformation", "TPC Sector Tracking", "TPC Track Merging and Fit", "TPC Compression", "TRD Tracking", "ITS Tracking", "TPC dEdx Computation", "TPC Cluster Finding", "TPC Decompression", "Global Refit"};
  static constexpr const char* const GENERAL_STEP_NAMES[] = {"Prepare", "QA"};
  typedef bitfield<RecoStep, unsigned int> RecoStepField;
  typedef bitfield<InOutType, unsigned int> InOutTypeField;
  constexpr static int N_RECO_STEPS = sizeof(GPUDataTypes::RECO_STEP_NAMES) / sizeof(GPUDataTypes::RECO_STEP_NAMES[0]);
  constexpr static int N_GENERAL_STEPS = sizeof(GPUDataTypes::GENERAL_STEP_NAMES) / sizeof(GPUDataTypes::GENERAL_STEP_NAMES[0]);
#endif
#ifdef GPUCA_NOCOMPAT
  static constexpr unsigned int NSLICES = 36;
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

template <class T>
struct DefaultPtr {
  typedef T type;
};
template <class T>
struct ConstPtr {
  typedef const T type;
};

template <template <typename T> class S>
struct GPUCalibObjectsTemplate {
  typename S<TPCFastTransform>::type* fastTransform = nullptr;
  typename S<o2::base::MatLayerCylSet>::type* matLUT = nullptr;
  typename S<o2::trd::GeometryFlat>::type* trdGeometry = nullptr;
  typename S<TPCdEdxCalibrationSplines>::type* dEdxSplines = nullptr;
  typename S<TPCPadGainCalib>::type* tpcPadGain = nullptr;
  typename S<o2::base::Propagator>::type* o2Propagator = nullptr;
};
typedef GPUCalibObjectsTemplate<DefaultPtr> GPUCalibObjects; // NOTE: These 2 must have identical layout since they are memcopied
typedef GPUCalibObjectsTemplate<ConstPtr> GPUCalibObjectsConst;

struct GPUTrackingInOutZS {
  static constexpr unsigned int NSLICES = GPUDataTypes::NSLICES;
  static constexpr unsigned int NENDPOINTS = 20;
  struct GPUTrackingInOutZSSlice {
    const void* const* zsPtr[NENDPOINTS];
    const unsigned int* nZSPtr[NENDPOINTS];
    unsigned int count[NENDPOINTS];
  };
  struct GPUTrackingInOutZSCounts {
    unsigned int count[NSLICES][NENDPOINTS] = {};
  };
  struct GPUTrackingInOutZSMeta {
    void* ptr[NSLICES][NENDPOINTS];
    unsigned int n[NSLICES][NENDPOINTS];
  };
  GPUTrackingInOutZSSlice slice[NSLICES];
};

struct GPUTrackingInOutDigits {
  static constexpr unsigned int NSLICES = GPUDataTypes::NSLICES;
  const o2::tpc::Digit* tpcDigits[NSLICES] = {nullptr};
  size_t nTPCDigits[NSLICES] = {0};
  GPUTPCDigitsMCInput* tpcDigitsMC;
};

struct GPUTrackingInOutPointers {
  GPUTrackingInOutPointers() = default;
  GPUTrackingInOutPointers(const GPUTrackingInOutPointers&) = default;
  static constexpr unsigned int NSLICES = GPUDataTypes::NSLICES;

  const GPUTrackingInOutZS* tpcZS = nullptr;
  const GPUTrackingInOutDigits* tpcPackedDigits = nullptr;
  const GPUTPCClusterData* clusterData[NSLICES] = {nullptr};
  unsigned int nClusterData[NSLICES] = {0};
  const AliHLTTPCRawCluster* rawClusters[NSLICES] = {nullptr};
  unsigned int nRawClusters[NSLICES] = {0};
  const o2::tpc::ClusterNativeAccess* clustersNative = nullptr;
  const GPUTPCTrack* sliceTracks[NSLICES] = {nullptr};
  unsigned int nSliceTracks[NSLICES] = {0};
  const GPUTPCHitId* sliceClusters[NSLICES] = {nullptr};
  unsigned int nSliceClusters[NSLICES] = {0};
  const AliHLTTPCClusterMCLabel* mcLabelsTPC = nullptr;
  unsigned int nMCLabelsTPC = 0;
  const GPUTPCMCInfo* mcInfosTPC = nullptr;
  unsigned int nMCInfosTPC = 0;
  const GPUTPCGMMergedTrack* mergedTracks = nullptr;
  unsigned int nMergedTracks = 0;
  const GPUTPCGMMergedTrackHit* mergedTrackHits = nullptr;
  const GPUTPCGMMergedTrackHitXYZ* mergedTrackHitsXYZ = nullptr;
  unsigned int nMergedTrackHits = 0;
  unsigned int* mergedTrackHitAttachment = nullptr;
  unsigned char* mergedTrackHitStates = nullptr;
  o2::tpc::TrackTPC* outputTracksTPCO2 = nullptr;
  unsigned int nOutputTracksTPCO2 = 0;
  unsigned int* outputClusRefsTPCO2 = nullptr;
  unsigned int nOutputClusRefsTPCO2 = 0;
  o2::MCCompLabel* outputTracksTPCO2MC = nullptr;
  const o2::tpc::CompressedClustersFlat* tpcCompressedClusters = nullptr;
  const GPUTRDTrackletWord* trdTracklets = nullptr;
  unsigned int nTRDTracklets = 0;
  const GPUTRDTrackletLabels* trdTrackletsMC = nullptr;
  unsigned int nTRDTrackletsMC = 0;
  const GPUTRDTrackGPU* trdTracks = nullptr;
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
