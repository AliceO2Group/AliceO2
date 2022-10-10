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
template <typename T>
class BaseCluster;
namespace base
{
template <typename T>
class PropagatorImpl;
class MatLayerCylSet;
} // namespace base
namespace track
{
#ifdef GPUCA_NOCOMPAT
template <typename value_T>
class TrackParametrizationWithError;
using TrackParCov = TrackParametrizationWithError<float>;
#else
class TrackParCov;
#endif
} // namespace track
namespace trd
{
class GeometryFlat;
} // namespace trd
namespace dataformats
{
class TrackTPCITS;
class MatchInfoTOF;
template <class T>
class MCTruthContainer;
template <class T>
class ConstMCTruthContainerView;
} // namespace dataformats
namespace itsmft
{
class CompClusterExt;
class ROFRecord;
class TopologyDictionary;
} // namespace itsmft
namespace its
{
class TrackITS;
} // namespace its
namespace tof
{
class Cluster;
} // namespace tof
namespace tpc
{
class CalibdEdxContainer;
} // namespace tpc
} // namespace o2

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class CorrectionMapsHelper;
class TPCFastTransform;
struct TPCPadGainCalib;
struct TPCZSLinkMapping;
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
class GPUTRDSpacePoint;
struct GPUTPCMCInfo;
struct GPUTPCMCInfoCol;
struct GPUTPCClusterData;
struct GPUTRDTrackletLabels;
struct GPUTPCDigitsMCInput;
struct GPUSettingsTF;

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
  static constexpr const char* const DEVICE_TYPE_NAMES[] = {"INVALID", "CPU", "CUDA", "HIP", "OCL", "OCL2"};
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
struct GPUCalibObjectsTemplate { // use only pointers on PODs or flat objects here
  typename S<TPCFastTransform>::type* fastTransform = nullptr;
  typename S<TPCFastTransform>::type* fastTransformRef = nullptr;
  typename S<CorrectionMapsHelper>::type* fastTransformHelper = nullptr;
  typename S<o2::base::MatLayerCylSet>::type* matLUT = nullptr;
  typename S<o2::trd::GeometryFlat>::type* trdGeometry = nullptr;
  typename S<TPCPadGainCalib>::type* tpcPadGain = nullptr;
  typename S<TPCZSLinkMapping>::type* tpcZSLinkMapping = nullptr;
  typename S<o2::tpc::CalibdEdxContainer>::type* dEdxCalibContainer = nullptr;
  typename S<o2::base::PropagatorImpl<float>>::type* o2Propagator = nullptr;
  typename S<o2::itsmft::TopologyDictionary>::type* itsPatternDict = nullptr;
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
  const GPUTPCDigitsMCInput* tpcDigitsMC = nullptr;
};

struct GPUTrackingInOutPointers {
  GPUTrackingInOutPointers() = default;

  // TPC
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
  const GPUTPCMCInfoCol* mcInfosTPCCol = nullptr;
  unsigned int nMCInfosTPCCol = 0;
  const GPUTPCGMMergedTrack* mergedTracks = nullptr;
  unsigned int nMergedTracks = 0;
  const GPUTPCGMMergedTrackHit* mergedTrackHits = nullptr;
  const GPUTPCGMMergedTrackHitXYZ* mergedTrackHitsXYZ = nullptr;
  unsigned int nMergedTrackHits = 0;
  const unsigned int* mergedTrackHitAttachment = nullptr;
  const unsigned char* mergedTrackHitStates = nullptr;
  const o2::tpc::TrackTPC* outputTracksTPCO2 = nullptr;
  unsigned int nOutputTracksTPCO2 = 0;
  const unsigned int* outputClusRefsTPCO2 = nullptr;
  unsigned int nOutputClusRefsTPCO2 = 0;
  const o2::MCCompLabel* outputTracksTPCO2MC = nullptr;
  const o2::tpc::CompressedClustersFlat* tpcCompressedClusters = nullptr;

  // TPC links
  int* tpcLinkITS = nullptr;
  int* tpcLinkTRD = nullptr;
  int* tpcLinkTOF = nullptr;
  const o2::track::TrackParCov** globalTracks = nullptr;
  float* globalTrackTimes = nullptr;
  unsigned int nGlobalTracks = 0;

  // TRD
  const GPUTRDTrackletWord* trdTracklets = nullptr;
  const GPUTRDSpacePoint* trdSpacePoints = nullptr;
  unsigned int nTRDTracklets = 0;
  const GPUTRDTrackGPU* trdTracks = nullptr;
  const GPUTRDTrack* trdTracksO2 = nullptr;
  unsigned int nTRDTracks = 0;
  const float* trdTriggerTimes = nullptr;
  const int* trdTrackletIdxFirst = nullptr;
  const char* trdTrigRecMask = nullptr;
  unsigned int nTRDTriggerRecords = 0;
  const GPUTRDTrack* trdTracksITSTPCTRD = nullptr;
  unsigned int nTRDTracksITSTPCTRD = 0;
  const GPUTRDTrack* trdTracksTPCTRD = nullptr;
  unsigned int nTRDTracksTPCTRD = 0;

  // TOF
  const o2::tof::Cluster* tofClusters = nullptr;
  unsigned int nTOFClusters = 0;
  const o2::dataformats::MatchInfoTOF* itstpctofMatches = nullptr;
  unsigned int nITSTPCTOFMatches = 0;
  const o2::dataformats::MatchInfoTOF* itstpctrdtofMatches = nullptr;
  unsigned int nITSTPCTRDTOFMatches = 0;
  const o2::dataformats::MatchInfoTOF* tpctrdtofMatches = nullptr;
  unsigned int nTPCTRDTOFMatches = 0;
  const o2::dataformats::MatchInfoTOF* tpctofMatches = nullptr;
  unsigned int nTPCTOFMatches = 0;

  // ITS
  const o2::itsmft::CompClusterExt* itsCompClusters = nullptr;
  const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* itsClusterMC = nullptr;
  const o2::BaseCluster<float>* itsClusters = nullptr;
  unsigned int nItsClusters = 0;
  const o2::itsmft::ROFRecord* itsClusterROF = nullptr;
  unsigned int nItsClusterROF = 0;
  const o2::its::TrackITS* itsTracks = nullptr;
  const o2::MCCompLabel* itsTrackMC = nullptr;
  unsigned int nItsTracks = 0;
  const int* itsTrackClusIdx = nullptr;
  const o2::itsmft::ROFRecord* itsTrackROF = nullptr;
  unsigned int nItsTrackROF = 0;

  // TPC-ITS
  const o2::dataformats::TrackTPCITS* tracksTPCITSO2 = nullptr;
  unsigned int nTracksTPCITSO2 = 0;

  // Common
  const GPUSettingsTF* settingsTF = nullptr;
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
