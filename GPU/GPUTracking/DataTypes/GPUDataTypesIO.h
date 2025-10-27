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

/// \file GPUDataTypesIO.h
/// \author David Rohr

#ifndef GPUDATATYPESIO_H
#define GPUDATATYPESIO_H

#include "GPUCommonDef.h"

// These are basic and non-complex data types, which will also be visible on the GPU.
// Please add complex data types required on the host but not GPU to GPUHostDataTypes.h and forward-declare!
#ifndef GPUCA_GPUCODE_DEVICE
#include <cstddef>
#endif
#include "GPUTRDDef.h"

struct AliHLTTPCClusterMCLabel;
struct AliHLTTPCRawCluster;
namespace o2::tpc
{
struct ClusterNativeAccess;
struct CompressedClustersFlat;
class Digit;
class TrackTPC;
namespace constants
{
} // namespace constants
} // namespace o2::tpc

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
template <typename value_T>
class TrackParametrizationWithError;
using TrackParCov = TrackParametrizationWithError<float>;
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
class ORTRootSerializer;
} // namespace tpc
} // namespace o2

namespace o2::gpu
{
class CorrectionMapsHelper;
class TPCFastTransformPOD;
struct TPCPadGainCalib;
struct TPCZSLinkMapping;

class GPUTRDRecoParam;
class GPUTPCTrack;
class GPUTPCHitId;
class GPUTPCGMMergedTrack;
struct GPUTPCGMMergedTrackHit;
class GPUTRDTrackletWord;
class GPUTRDSpacePoint;
struct GPUTPCMCInfo;
struct GPUTPCMCInfoCol;
struct GPUTPCClusterData;
struct GPUTRDTrackletLabels;
struct GPUTPCDigitsMCInput;
struct GPUSettingsTF;

namespace gpudatatypes
{
static constexpr uint32_t NSECTORS = 36;
} // namespace gpudatatypes

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
  typename S<TPCFastTransformPOD>::type* fastTransform = nullptr;
  typename S<CorrectionMapsHelper>::type* fastTransformHelper = nullptr;
  typename S<o2::base::MatLayerCylSet>::type* matLUT = nullptr;
  typename S<o2::trd::GeometryFlat>::type* trdGeometry = nullptr;
  typename S<TPCPadGainCalib>::type* tpcPadGain = nullptr;
  typename S<TPCZSLinkMapping>::type* tpcZSLinkMapping = nullptr;
  typename S<o2::tpc::CalibdEdxContainer>::type* dEdxCalibContainer = nullptr;
  typename S<o2::base::PropagatorImpl<float>>::type* o2Propagator = nullptr;
  typename S<o2::itsmft::TopologyDictionary>::type* itsPatternDict = nullptr;
  typename S<GPUTRDRecoParam>::type* trdRecoParam = nullptr;
  // NN clusterizer objects
  typename S<o2::tpc::ORTRootSerializer>::type* nnClusterizerNetworks[3] = {nullptr, nullptr, nullptr};
};
typedef GPUCalibObjectsTemplate<DefaultPtr> GPUCalibObjects; // NOTE: These 2 must have identical layout since they are memcopied
typedef GPUCalibObjectsTemplate<ConstPtr> GPUCalibObjectsConst;

struct GPUTrackingInOutZS {
  static constexpr uint32_t NSECTORS = gpudatatypes::NSECTORS;
  static constexpr uint32_t NENDPOINTS = 20;
  struct GPUTrackingInOutZSSector {
    const void* const* zsPtr[NENDPOINTS];
    const uint32_t* nZSPtr[NENDPOINTS];
    uint32_t count[NENDPOINTS];
  };
  struct GPUTrackingInOutZSCounts {
    uint32_t count[NSECTORS][NENDPOINTS] = {};
  };
  struct GPUTrackingInOutZSMeta {
    void* ptr[NSECTORS][NENDPOINTS];
    uint32_t n[NSECTORS][NENDPOINTS];
  };
  GPUTrackingInOutZSSector sector[NSECTORS];
};

struct GPUTrackingInOutDigits {
  static constexpr uint32_t NSECTORS = gpudatatypes::NSECTORS;
  const o2::tpc::Digit* tpcDigits[NSECTORS] = {nullptr};
  size_t nTPCDigits[NSECTORS] = {0};
  const GPUTPCDigitsMCInput* tpcDigitsMC = nullptr;
};

struct GPUTrackingInOutPointers {
  GPUTrackingInOutPointers() = default;

  // TPC
  static constexpr uint32_t NSECTORS = gpudatatypes::NSECTORS;
  const GPUTrackingInOutZS* tpcZS = nullptr;
  const GPUTrackingInOutDigits* tpcPackedDigits = nullptr;
  const GPUTPCClusterData* clusterData[NSECTORS] = {nullptr};
  uint32_t nClusterData[NSECTORS] = {0};
  const AliHLTTPCRawCluster* rawClusters[NSECTORS] = {nullptr};
  uint32_t nRawClusters[NSECTORS] = {0};
  const o2::tpc::ClusterNativeAccess* clustersNative = nullptr;
  const GPUTPCTrack* sectorTracks[NSECTORS] = {nullptr};
  uint32_t nSectorTracks[NSECTORS] = {0};
  const GPUTPCHitId* sectorClusters[NSECTORS] = {nullptr};
  uint32_t nSectorClusters[NSECTORS] = {0};
  const AliHLTTPCClusterMCLabel* mcLabelsTPC = nullptr;
  uint32_t nMCLabelsTPC = 0;
  const GPUTPCMCInfo* mcInfosTPC = nullptr;
  uint32_t nMCInfosTPC = 0;
  const GPUTPCMCInfoCol* mcInfosTPCCol = nullptr;
  uint32_t nMCInfosTPCCol = 0;
  const GPUTPCGMMergedTrack* mergedTracks = nullptr;
  uint32_t nMergedTracks = 0;
  const GPUTPCGMMergedTrackHit* mergedTrackHits = nullptr;
  uint32_t nMergedTrackHits = 0;
  const uint32_t* mergedTrackHitAttachment = nullptr;
  const uint8_t* mergedTrackHitStates = nullptr;
  const o2::tpc::TrackTPC* outputTracksTPCO2 = nullptr;
  uint32_t nOutputTracksTPCO2 = 0;
  const uint32_t* outputClusRefsTPCO2 = nullptr;
  uint32_t nOutputClusRefsTPCO2 = 0;
  const o2::MCCompLabel* outputTracksTPCO2MC = nullptr;
  const o2::tpc::CompressedClustersFlat* tpcCompressedClusters = nullptr;
  const o2::tpc::ClusterNativeAccess* clustersNativeReduced = nullptr;

  // TPC links
  int32_t* tpcLinkITS = nullptr;
  int32_t* tpcLinkTRD = nullptr;
  int32_t* tpcLinkTOF = nullptr;
  const o2::track::TrackParCov** globalTracks = nullptr;
  float* globalTrackTimes = nullptr;
  uint32_t nGlobalTracks = 0;

  // TRD
  const GPUTRDTrackletWord* trdTracklets = nullptr;
  const GPUTRDSpacePoint* trdSpacePoints = nullptr;
  uint32_t nTRDTracklets = 0;
  const GPUTRDTrackGPU* trdTracks = nullptr;
  const GPUTRDTrack* trdTracksO2 = nullptr;
  uint32_t nTRDTracks = 0;
  const float* trdTriggerTimes = nullptr;
  const int32_t* trdTrackletIdxFirst = nullptr;
  const uint8_t* trdTrigRecMask = nullptr;
  uint32_t nTRDTriggerRecords = 0;
  const GPUTRDTrack* trdTracksITSTPCTRD = nullptr;
  uint32_t nTRDTracksITSTPCTRD = 0;
  const GPUTRDTrack* trdTracksTPCTRD = nullptr;
  uint32_t nTRDTracksTPCTRD = 0;

  // TOF
  const o2::tof::Cluster* tofClusters = nullptr;
  uint32_t nTOFClusters = 0;
  const o2::dataformats::MatchInfoTOF* itstpctofMatches = nullptr;
  uint32_t nITSTPCTOFMatches = 0;
  const o2::dataformats::MatchInfoTOF* itstpctrdtofMatches = nullptr;
  uint32_t nITSTPCTRDTOFMatches = 0;
  const o2::dataformats::MatchInfoTOF* tpctrdtofMatches = nullptr;
  uint32_t nTPCTRDTOFMatches = 0;
  const o2::dataformats::MatchInfoTOF* tpctofMatches = nullptr;
  uint32_t nTPCTOFMatches = 0;

  // ITS
  const o2::itsmft::CompClusterExt* itsCompClusters = nullptr;
  const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* itsClusterMC = nullptr;
  const o2::BaseCluster<float>* itsClusters = nullptr;
  uint32_t nItsClusters = 0;
  const o2::itsmft::ROFRecord* itsClusterROF = nullptr;
  uint32_t nItsClusterROF = 0;
  const o2::its::TrackITS* itsTracks = nullptr;
  const o2::MCCompLabel* itsTrackMC = nullptr;
  uint32_t nItsTracks = 0;
  const int32_t* itsTrackClusIdx = nullptr;
  const o2::itsmft::ROFRecord* itsTrackROF = nullptr;
  uint32_t nItsTrackROF = 0;

  // TPC-ITS
  const o2::dataformats::TrackTPCITS* tracksTPCITSO2 = nullptr;
  uint32_t nTracksTPCITSO2 = 0;

  // Common
  const GPUSettingsTF* settingsTF = nullptr;
};

} // namespace o2::gpu

#endif
