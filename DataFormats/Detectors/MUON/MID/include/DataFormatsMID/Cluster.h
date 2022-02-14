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

/// \file   DataFormatsMID/Cluster.h
/// \brief  Reconstructed MID cluster
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   08 December 2021

#ifndef O2_MID_CLUSTER_H
#define O2_MID_CLUSTER_H

#include <ostream>
#include <cstdint>

#include "Rtypes.h"

namespace o2
{
namespace mid
{
///  cluster structure for MID
struct Cluster {
  float xCoor = 0;           ///< x coordinate
  float yCoor = 0;           ///< y coordinate
  float zCoor = 0;           ///< z coordinate
  float xErr = 0;            ///< Cluster resolution along x
  float yErr = 0;            ///< Cluster resolution along y
  uint8_t deId = 0;          ///< Detection element ID
  uint8_t firedCathodes = 0; ///< Fired cathodes

  /// Sets a flag specifying if the BP or NBP were fired
  /// \param cathode can be 0 (bending-plane) or 1 (non-bending plane)
  /// \param isFired true if it was fired
  void setFired(int cathode, bool isFired = true);

  /// Sets a flag specifying that both BP and NBP were fired
  inline void setBothFired() { firedCathodes = 0x3; }

  /// Returns the x position
  inline double getX() const { return xCoor; }

  /// Returns the y position
  inline double getY() const { return yCoor; }

  /// Returns the z position
  inline double getZ() const { return zCoor; }

  /// Returns the x resolution
  inline double getEX() const { return xErr; }

  /// Returns the square of the x resolution
  inline double getEX2() const { return getEX() * getEX(); }

  /// Returns the y resolution
  inline double getEY() const { return yErr; }

  /// Returns the square of the y resolution
  inline double getEY2() const { return getEY() * getEY(); }

  /// Checks if cathode was fired
  bool isFired(int cathode) const { return (firedCathodes >> cathode) & 0x1; }

  ClassDefNV(Cluster, 1);
};

std::ostream& operator<<(std::ostream& os, const Cluster& data);

} // namespace mid
} // namespace o2

#endif /* O2_MID_CLUSTER_H */
