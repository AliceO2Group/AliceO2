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
#ifndef __OPENCL__
#include <cstdint>
#include <cstddef>   // for size_t
#endif
#include "DataFormatsTPC/Constants.h"
#include "GPUCommonDef.h"

namespace o2
{
class MCCompLabel;
namespace dataformats
{
template <class T>
class MCTruthContainer;
}
}

namespace o2
{
namespace TPC
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
  static constexpr int scaleTimePacked =
    64; //< ~50 is needed for 0.1mm precision, but leads to float rounding artifacts around 20ms
  static constexpr int scalePadPacked = 64; //< ~60 is needed for 0.1mm precision, but power of two avoids rounding
  static constexpr int scaleSigmaTimePacked = 32; // 1/32nd of pad/timebin precision for cluster size
  static constexpr int scaleSigmaPadPacked = 32;

  uint32_t timeFlagsPacked; //< Contains the time in the lower 24 bits in a packed format, contains the flags in the
                            // upper 8 bits
  uint16_t padPacked;       //< Contains the pad in a packed format
  uint8_t sigmaTimePacked;  //< Sigma of the time in packed format
  uint8_t sigmaPadPacked;   //< Sigma of the pad in packed format
  uint16_t qMax;            //< QMax of the cluster
  uint16_t qTot;            //< Total charge of the cluster

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
  GPUd() float getTime() const { return float(timeFlagsPacked & 0xFFFFFF) / scaleTimePacked; }
  GPUd() void setTime(float time)
  {
    timeFlagsPacked =
      (((decltype(timeFlagsPacked))(time * scaleTimePacked + 0.5)) & 0xFFFFFF) | (timeFlagsPacked & 0xFF000000);
  }
  GPUd() void setTimeFlags(float time, uint8_t flags)
  {
    timeFlagsPacked = (((decltype(timeFlagsPacked))(time * scaleTimePacked + 0.5)) & 0xFFFFFF) |
                      ((decltype(timeFlagsPacked))flags << 24);
  }
  GPUd() float getPad() const { return float(padPacked) / scalePadPacked; }
  GPUd() void setPad(float pad) { padPacked = (decltype(padPacked))(pad * scalePadPacked + 0.5); }
  GPUd() float getSigmaTime() const { return float(sigmaTimePacked) / scaleSigmaTimePacked; }
  GPUd() void setSigmaTime(float sigmaTime)
  {
    uint32_t tmp = sigmaTime * scaleSigmaTimePacked + 0.5;
    if (tmp > 0xFF) {
      tmp = 0xFF;
    }
    sigmaTimePacked = tmp;
  }
  GPUd() float getSigmaPad() const { return float(sigmaPadPacked) / scaleSigmaPadPacked; }
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
    } else {
      return (this->padPacked < rhs.padPacked);
    }
  }
};

// This is an index struct to access TPC clusters inside sectors and rows. It shall not own the data, but just point to
// the data inside a buffer.
struct ClusterNativeAccessFullTPC {
  const ClusterNative* clusters[o2::TPC::Constants::MAXSECTOR][o2::TPC::Constants::MAXGLOBALPADROW];
  unsigned int nClusters[o2::TPC::Constants::MAXSECTOR][o2::TPC::Constants::MAXGLOBALPADROW];
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clustersMCTruth[o2::TPC::Constants::MAXSECTOR]
                                                                     [o2::TPC::Constants::MAXGLOBALPADROW];
};
}
}
#endif
