// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   DataFormatsMID/Cluster3D.h
/// \brief  Reconstructed MID cluster (global coordinates)
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   31 August 2017

#ifndef O2_MID_CLUSTER3D_H
#define O2_MID_CLUSTER3D_H

#include <ostream>
#include <cstdint>

namespace o2
{
namespace mid
{
/// 3D cluster structure for MID
struct Cluster3D {
  uint8_t deId;  ///< Index of the detection element
  float xCoor;   ///< x coordinate
  float yCoor;   ///< y coordinate
  float zCoor;   ///< z coordinate
  float sigmaX2; ///< Dispersion along x
  float sigmaY2; ///< Dispersion along y
};

inline std::ostream& operator<<(std::ostream& os, const Cluster3D& data)
{
  /// Overload ostream operator
  os << "deId: " << static_cast<int>(data.deId) << "  position: (" << data.xCoor << ", " << data.yCoor << ", " << data.zCoor << ")  variance: (" << data.sigmaX2 << ", " << data.sigmaY2 << ")";
  return os;
}

} // namespace mid
} // namespace o2

#endif /* O2_MID_CLUSTER3D_H */
