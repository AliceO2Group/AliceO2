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

namespace o2
{
namespace mch
{

/// cluster minimal structure
struct ClusterStruct {
  float x;      ///< cluster position along x
  float y;      ///< cluster position along y
  float z;      ///< cluster position along z
  float ex;     ///< cluster resolution along x
  float ey;     ///< cluster resolution along y
  uint32_t uid; ///< cluster unique ID

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
};

std::ostream& operator<<(std::ostream& stream, const ClusterStruct& cluster);

} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_CLUSTERBLOCK_H_
