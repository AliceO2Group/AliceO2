// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   DataFormatsMID/Cluster2D.h
/// \brief  Reconstructed cluster per RPC
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   30 August 2017

#ifndef O2_MID_CLUSTER2D_H
#define O2_MID_CLUSTER2D_H

#include <ostream>
#include <cstdint>

namespace o2
{
namespace mid
{
/// 2D cluster structure for MID
struct Cluster2D {
  uint8_t deId;  ///< Detection element ID
  float xCoor;   ///< Local x coordinate
  float yCoor;   ///< Local y coordinate
  float sigmaX2; ///< Square of dispersion along x
  float sigmaY2; ///< Square of dispersion along y
};

inline std::ostream& operator<<(std::ostream& os, const Cluster2D& data)
{
  /// Overload ostream operator
  os << "deId: " << static_cast<int>(data.deId) << "  position: (" << data.xCoor << ", " << data.yCoor << ")  variance: (" << data.sigmaX2 << ", " << data.sigmaY2 << ")";
  return os;
}
} // namespace mid
} // namespace o2

#endif /* O2_MID_CLUSTER2D_H */
