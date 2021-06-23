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

#ifndef CLUSTERGROUPATTRIBUTE_H
#define CLUSTERGROUPATTRIBUTE_H

/// \file ClusterGroupAttribute.h
/// \brief Meta data for a group describing it by sector number and global padrow
/// \since 2018-04-17

#include <cstdint>

namespace o2
{
namespace tpc
{

/**
 * \struct ClusterGroupAttribute
 * Meta data attribute for a group of Cluster objects
 */
struct ClusterGroupAttribute {
  uint8_t sector;
  uint8_t globalPadRow;

  void set(uint32_t subSpecification)
  {
    sector = (subSpecification >> 16) & 0xff;
    globalPadRow = subSpecification & 0xff;
  }

  uint32_t getSubSpecification() const { return sector << 16 | globalPadRow; }
};

/**
 * \struct ClusterGroupHeader
 * Group attribute extended by number-of-clusters member.
 * This struct is intended to be sent as part of unserialized data packets
 */
struct ClusterGroupHeader : public ClusterGroupAttribute {
  uint16_t nClusters;

  ClusterGroupHeader(const ClusterGroupAttribute& attr, uint16_t n) : ClusterGroupAttribute(attr), nClusters(n) {}
};

static_assert(sizeof(ClusterGroupAttribute) == 2, "inconsistent padding detected");
static_assert(sizeof(ClusterGroupHeader) == 4, "inconsistent padding detected");

} // namespace tpc
} // namespace o2

#endif // CLUSTERGROUPATTRIBUTE_H
