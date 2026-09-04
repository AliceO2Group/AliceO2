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

/// \file Cluster.cxx
/// \brief Implementation of the IOTOF cluster

#include "DataFormatsIOTOF/Cluster.h"
#include "Framework/Logger.h"
#include <cassert>
#include <iostream>
#include <format>

// Root ClassImp macros for serialization metadata
ClassImp(o2::iotof::ClusterInfo);
ClassImp(o2::iotof::Cluster);

namespace o2
{
namespace iotof
{

std::string Cluster::asString() const
{
  LOG(debug) << "[Cluster::asString] Converting Cluster to string";
  return std::format(
    "chip: {:5d} | row: {:3d} col: {:3d} | span: {:2d}x{:2d} | pattern: {:5d} topology: {:4d}",
    getChipID(),
    getRow(),
    getCol(),
    getRowSpan(),
    getColSpan(),
    getPattern(),
    getTopology()
  );
}

//______________________________________________________________________________
void Cluster::print() const
{
  std::cout << *this << "\n";
}

//______________________________________________________________________________
void Cluster::sanityCheck()
{
  LOG(debug) << "[Cluster::sanityCheck] Performing sanity check on Cluster fields";

  // Ensure extracted values fit within allowed bit masks
  assert(getRow()      <= ClusterInfo::MaskRow);
  assert(getCol()      <= ClusterInfo::MaskCol);
  assert(getRowSpan()  <= ClusterInfo::MaskRowSpan);
  assert(getColSpan()  <= ClusterInfo::MaskColSpan);
  assert(getPattern()  <= ClusterInfo::MaskPattern);
  assert(getTopology() <= ClusterInfo::MaskTopology);
}

} // namespace iotof
} // namespace o2

// Stream operator implementation
std::ostream& operator<<(std::ostream& stream, const o2::iotof::Cluster& cl)
{
  stream << cl.asString();
  return stream;
}
