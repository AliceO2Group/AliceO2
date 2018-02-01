// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterBlock.cxx
/// \brief Implementation of the MCH cluster minimal structure
///
/// \author Philippe Pillot, Subatech

#include "MCHBase/ClusterBlock.h"

namespace o2
{
namespace mch
{

//_________________________________________________________________________
std::ostream& operator<<(std::ostream& stream, const ClusterStruct& cluster)
{
  auto oldflags = stream.flags();
  stream << "{x = " << cluster.x << ", y = " << cluster.y << ", z = " << cluster.z << ", ex = " << cluster.ex
         << ", ey = " << cluster.ey << ", uid = " << cluster.uid << "}";
  stream.flags(oldflags);
  return stream;
}

} // namespace mch
} // namespace o2
