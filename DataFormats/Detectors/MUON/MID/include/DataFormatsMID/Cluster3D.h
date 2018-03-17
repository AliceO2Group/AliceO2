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

#include <cstdint>
#include "MathUtils/Cartesian3D.h"

namespace o2
{
namespace mid
{
/// 3D cluster structure for MID
struct Cluster3D {
  uint16_t id;             ///< Unique Id of the cluster in the detection element
  uint8_t deId;            ///< Index of the detection element
  Point3D<float> position; ///< Global position
  float sigmaX2;           ///< Dispersion along x
  float sigmaY2;           ///< Dispersion along y
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_CLUSTER3D_H */
