// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsTRK/Cluster.h"
#include <sstream>

ClassImp(o2::trk::Cluster);

namespace o2::trk
{

std::string Cluster::asString() const
{
  std::ostringstream stream;
  stream << "chip=" << chipID << " row=" << row << " col=" << col << " size=" << size
         << " subDet=" << subDetID << " layer=" << layer << " disk=" << disk;
  return stream.str();
}

} // namespace o2::trk
