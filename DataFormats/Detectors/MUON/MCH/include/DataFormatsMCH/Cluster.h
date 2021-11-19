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

/// \file Cluster.h
/// \brief Definition of the MCH cluster minimal structure
///
/// \author Philippe Pillot, Subatech

#ifndef ALICEO2_MCH_CLUSTER_H_
#define ALICEO2_MCH_CLUSTER_H_

#include <iostream>
#include <stdexcept>
#include <string>
#include <Rtypes.h>

namespace o2
{
namespace mch
{

/// cluster minimal structure
struct Cluster {
  float x;             ///< cluster position along x
  float y;             ///< cluster position along y
  float z;             ///< cluster position along z
  float ex;            ///< cluster resolution along x
  float ey;            ///< cluster resolution along y
  uint32_t uid;        ///< cluster unique ID
  uint32_t firstDigit; ///< index of first associated digit in the ordered vector of digits
  uint32_t nDigits;    ///< number of digits attached to this cluster

  /// Return the cluster position along x as double
  double getX() const { return x; }
  /// Return the cluster position along y as double
  double getY() const { return y; }
  /// Return the cluster position along z as double
  double getZ() const { return z; }
  /// Return the cluster resolution along x as double
  double getEx() const { return ex; }
  /// Return the cluster resolution along y as double
  double getEy() const { return ey; }
  /// Return the cluster resolution square along x
  double getEx2() const { return getEx() * getEx(); }
  /// Return the cluster resolution square along y
  double getEy2() const { return getEy() * getEy(); }

  /// Return the unique ID of this cluster in human readable form
  std::string getIdAsString() const { return "DE" + std::to_string(getDEId()) + "#" + std::to_string(getClusterIndex()); }

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

  ClassDefNV(Cluster, 1)
};

std::ostream& operator<<(std::ostream& stream, const Cluster& cluster);
} // namespace mch
} // namespace o2

namespace framework
{
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::mch::Cluster> : std::true_type {
};
} // namespace framework

#endif // ALICEO2_MCH_CLUSTER_H_
