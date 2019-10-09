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

/// \file   MID/src/Cluster.cxx
/// \brief  Reconstructed MID cluster
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   08 December 2021

#include "DataFormatsMID/Cluster.h"
namespace o2
{
namespace mid
{
void Cluster::setFired(int cathode, bool isFired)
{
  firedCathodes = (firedCathodes & ~(1 << cathode)) | (isFired << cathode);
}

std::ostream& operator<<(std::ostream& os, const Cluster& data)
{
  /// Overload ostream operator
  os << "deId: " << static_cast<int>(data.deId) << "  position: (" << data.xCoor << ", " << data.yCoor << ", " << data.zCoor << ")  resolution: (" << data.xErr << ", " << data.yErr << ")  fired BP: " << data.isFired(0) << "  NBP: " << data.isFired(1);
  return os;
}

} // namespace mid
} // namespace o2
