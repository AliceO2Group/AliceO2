// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterNative.h
/// \brief Class of a TPC cluster in TPC-native coordinates (row, time)
/// \author David Rohr
#ifndef ALICEO2_DATAFORMATSTPC_CLUSTERNATIVE_H
#define ALICEO2_DATAFORMATSTPC_CLUSTERNATIVE_H
#ifndef GPUCA_GPUCODE_DEVICE
#include <cstdint>
#include <cstddef> // for size_t
#include <utility>
#endif
#include "DataFormatsTPC/Constants.h"
#include "GPUCommonDef.h"

namespace o2
{
class MCCompLabel;
namespace dataformats
{
template <class T>
class ConstMCTruthContainer;
template <class T>
class ConstMCTruthContainerView;
}
} // namespace o2

namespace o2
{
namespace tpc
{
/**
 * \struct ClusterNative
 * A transient data structure for clusters in TPC-native pad,
 * and time format. To keep it as small as possible, row coordinate is
 * kept outside in the meta information for a sequence of ClusterNative
 * objects.
 *
 * Structure holds float values in a packed integral format, scaling
 * factors are chosen according to TPC resolution. The 24-bit wide time
 * field allows unique values within 512 TPC drift lengths.
 *
 * Not for permanent storage.
 */
struct ClusterNative {
  // NOTE: These states must match those from GPUTPCGMMergedTrackHit!
  enum clusterState { flagSplitPad = 0x1,  // Split in pad direction
                      flagSplitTime = 0x2, // Split in time direction
                      flagEdge = 0x4,      // At edge of TPC sector
                      flagSingle = 0x8 };  // Single pad or single time-bin cluster

  static constexpr int scaleTimePacked = 64;      //< ~50 is needed for 0.1mm precision, but leads to float rounding artifacts around 20ms
  static constexpr int scalePadPacked = 64;       //< ~60 is needed for 0.1mm precision, but power of two avoids rounding
  static constexpr int scaleSigmaTimePacked = 32; // 1/32nd of pad/timebin precision for cluster size
  static constexpr int scaleSigmaPadPacked = 32;

  uint32_t timeFlagsPacked; //< Contains the time in the lower 24 bits in a packed format, contains the flags in the
                            // upper 8 bits
  uint16_t padPacked;       //< Contains the pad in a packed format
  uint8_t sigmaTimePacked;  //< Sigma of the time in packed format
  uint8_t sigmaPadPacked;   //< Sigma of the pad in packed format
  uint16_t qMax;            //< QMax of the cluster
  uint16_t qTot;            //< Total charge of the cluster

  GPUd() static uint16_t packPad(float pad) { return (uint16_t)(pad * scalePadPacked + 0.5); }
  GPUd() static uint32_t packTime(float time) { return (uint32_t)(time * scaleTimePacked + 0.5); }
  GPUd() static float unpackPad(uint16_t pad) { return float(pad) * (1.f / scalePadPacked); }
  GPUd() static float unpackTime(uint32_t time) { return float(time) * (1.f / scaleTimePacked); }

  GPUdDefault() ClusterNative() CON_DEFAULT;
  GPUd() ClusterNative(uint32_t time, uint8_t flags, uint16_t pad, uint8_t sigmaTime, uint8_t sigmaPad, uint16_t qmax, uint16_t qtot) : padPacked(pad), sigmaTimePacked(sigmaTime), sigmaPadPacked(sigmaPad), qMax(qmax), qTot(qtot)
  {
    setTimePackedFlags(time, flags);
  }

  GPUd() uint8_t getFlags() const { return timeFlagsPacked >> 24; }
  GPUd() uint32_t getTimePacked() const { return timeFlagsPacked & 0xFFFFFF; }
  GPUd() void setTimePackedFlags(uint32_t timePacked, uint8_t flags)
  {
    timeFlagsPacked = (timePacked & 0xFFFFFF) | (uint32_t)flags << 24;
  }
  GPUd() void setTimePacked(uint32_t timePacked)
  {
    timeFlagsPacked = (timePacked & 0xFFFFFF) | (timeFlagsPacked & 0xFF000000);
  }
  GPUd() void setFlags(uint8_t flags) { timeFlagsPacked = (timeFlagsPacked & 0xFFFFFF) | ((uint32_t)flags << 24); }
  GPUd() float getTime() const { return unpackTime(timeFlagsPacked & 0xFFFFFF); }
  GPUd() void setTime(float time)
  {
    timeFlagsPacked = (packTime(time) & 0xFFFFFF) | (timeFlagsPacked & 0xFF000000);
  }
  GPUd() void setTimeFlags(float time, uint8_t flags)
  {
    timeFlagsPacked = (packTime(time) & 0xFFFFFF) | ((decltype(timeFlagsPacked))flags << 24);
  }

  /// @return Returns the pad position of the cluster.
  /// note that the pad position is defined on the left side of the pad.
  /// the pad position from clusters are calculated in HwClusterer::hwClusterProcessor()
  /// around the centre of gravity around the left side of the pad.
  /// i.e. the center of the first pad has pad position zero.
  /// To get the corresponding local Y coordinate of the cluster:
  /// Y = (pad_position - 0.5 * (n_pads - 1)) * padWidth
  /// example:
  /// the pad position is for example 12.4 (pad_position = 12.4).
  /// there are 66 pads in the first pad row (n_pads = 66).
  /// the pad width for pads in the first padrow is 4.16mm (padWidth = 4.16mm).
  /// Y = (12.4 - 0.5 * (66 - 1)) * 4.16mm = -83.616mm
  GPUd() float getPad() const { return unpackPad(padPacked); }
  GPUd() void setPad(float pad) { padPacked = packPad(pad); }
  GPUd() float getSigmaTime() const { return float(sigmaTimePacked) * (1.f / scaleSigmaTimePacked); }
  GPUd() void setSigmaTime(float sigmaTime)
  {
    uint32_t tmp = sigmaTime * scaleSigmaTimePacked + 0.5;
    if (tmp > 0xFF) {
      tmp = 0xFF;
    }
    sigmaTimePacked = tmp;
  }
  GPUd() float getSigmaPad() const { return float(sigmaPadPacked) * (1.f / scaleSigmaPadPacked); }
  GPUd() void setSigmaPad(float sigmaPad)
  {
    uint32_t tmp = sigmaPad * scaleSigmaPadPacked + 0.5;
    if (tmp > 0xFF) {
      tmp = 0xFF;
    }
    sigmaPadPacked = tmp;
  }

  GPUd() bool operator<(const ClusterNative& rhs) const
  {
    if (this->getTimePacked() != rhs.getTimePacked()) {
      return (this->getTimePacked() < rhs.getTimePacked());
    } else if (this->padPacked != rhs.padPacked) {
      return (this->padPacked < rhs.padPacked);
    } else if (this->sigmaTimePacked != rhs.sigmaTimePacked) {
      return (this->sigmaTimePacked < rhs.sigmaTimePacked);
    } else if (this->sigmaPadPacked != rhs.sigmaPadPacked) {
      return (this->sigmaPadPacked < rhs.sigmaPadPacked);
    } else if (this->qMax != rhs.qMax) {
      return (this->qMax < rhs.qMax);
    } else if (this->qTot != rhs.qTot) {
      return (this->qTot < rhs.qTot);
    } else {
      return (this->getFlags() < rhs.getFlags());
    }
  }
};

// This is an index struct to access TPC clusters inside sectors and rows. It shall not own the data, but just point to
// the data inside a buffer.
struct ClusterNativeAccess {
  const ClusterNative* clustersLinear;
  const ClusterNative* clusters[constants::MAXSECTOR][constants::MAXGLOBALPADROW];
  const o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel>* clustersMCTruth;
  unsigned int nClusters[constants::MAXSECTOR][constants::MAXGLOBALPADROW];
  unsigned int nClustersSector[constants::MAXSECTOR];
  unsigned int clusterOffset[constants::MAXSECTOR][constants::MAXGLOBALPADROW];
  unsigned int nClustersTotal;

  void setOffsetPtrs();

#ifndef GPUCA_GPUCODE
  using ConstMCLabelContainer = o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>;
  using ConstMCLabelContainerView = o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel>;
  using ConstMCLabelContainerViewWithBuffer = std::pair<ConstMCLabelContainer, ConstMCLabelContainerView>;
#endif
};

inline void ClusterNativeAccess::setOffsetPtrs()
{
  int offset = 0;
  for (unsigned int i = 0; i < constants::MAXSECTOR; i++) {
    nClustersSector[i] = 0;
    for (unsigned int j = 0; j < constants::MAXGLOBALPADROW; j++) {
      clusterOffset[i][j] = offset;
      clusters[i][j] = &clustersLinear[offset];
      nClustersSector[i] += nClusters[i][j];
      offset += nClusters[i][j];
    }
  }
  nClustersTotal = offset;
}
} // namespace tpc
} // namespace o2
#endif
