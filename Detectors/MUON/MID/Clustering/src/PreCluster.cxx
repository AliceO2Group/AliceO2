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

/// \file   MID/Clustering/src/PreCluster.cxx
/// \brief  Implementation of the pre-cluster for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   24 April 2019
#include "MIDClustering/PreCluster.h"

namespace o2
{
namespace mid
{

std::ostream& operator<<(std::ostream& os, const PreCluster& data)
{
  /// Output streamer for PreClusterNBP
  os << "deId: " << static_cast<int>(data.deId);
  os << "  cathode: " << static_cast<int>(data.cathode);
  os << "  (column, line, strip) first: (" << static_cast<int>(data.firstColumn) << ", " << static_cast<int>(data.firstLine) << ", " << static_cast<int>(data.firstStrip) << ")";
  os << "  last: (" << static_cast<int>(data.lastColumn) << ", " << static_cast<int>(data.lastLine) << ", " << static_cast<int>(data.lastStrip) << ")";
  return os;
}

} // namespace mid
} // namespace o2
