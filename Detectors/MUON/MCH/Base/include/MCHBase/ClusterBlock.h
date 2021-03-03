// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterBlock.h
/// \brief Definition of the MCH cluster minimal structure
///
/// \author Philippe Pillot, Subatech

#ifndef ALICEO2_MCH_CLUSTERBLOCK_H_
#define ALICEO2_MCH_CLUSTERBLOCK_H_

#include <iostream>
#include <stdexcept>

namespace o2
{
namespace mch
{

/// cluster minimal structure
struct ClusterStruct {
  float x;             ///< cluster position along x
  float y;             ///< cluster position along y
  float z;             ///< cluster position along z
  float ex;            ///< cluster resolution along x
  float ey;            ///< cluster resolution along y
  uint32_t uid;        ///< cluster unique ID
  uint32_t firstDigit; ///< index of first associated digit in the ordered vector of digits
  uint32_t nDigits;    ///< number of digits attached to this cluster

  /// Return the chamber ID (0..), part of the unique ID
  int getChamberId() const { return getChamberId(uid); }
  /// Return the detection element ID, part of the unique ID
  int getDEId() const { return getDEId(uid); }
  /// Return the index of this cluster (0..), part of the unique ID
  int getClusterIndex() const { return getClusterIndex(uid); }

  /// Return the chamber ID of the cluster, part of its unique ID
  static int getChamberId(uint32_t clusterId) { return (clusterId & 0xF0000000) >> 28; }
  /// Return the detection element ID of the cluster, part of its unique ID
  static int getDEId(uint32_t clusterId) { return (clusterId & 0x0FFE0000) >> 17; }
  /// Return the index of the cluster, part of its unique ID
  static int getClusterIndex(uint32_t clusterId) { return (clusterId & 0x0001FFFF); }

  /// Build the unique ID of the cluster from the chamber ID, detection element ID and cluster index
  static uint32_t buildUniqueId(int chamberId, int deId, int clusterIndex)
  {
    if ((clusterIndex & 0x1FFFF) != clusterIndex) {
      throw std::runtime_error("invalid cluster index. Cannot build unique ID");
    }
    return (((chamberId & 0xF) << 28) | ((deId & 0x7FF) << 17) | clusterIndex);
  }
};

std::ostream& operator<<(std::ostream& stream, const ClusterStruct& cluster);

} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_CLUSTERBLOCK_H_
