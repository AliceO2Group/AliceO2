// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDClustering/PreCluster.h
/// \brief  Pre-cluster structure for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   24 April 2019

#ifndef O2_MID_PRECLUSTER_H
#define O2_MID_PRECLUSTER_H

#include <cstdint>
#include <ostream>

namespace o2
{
namespace mid
{

struct PreCluster {
  uint8_t deId;        ///< Detection element ID
  uint8_t cathode;     ///< Cathode
  uint8_t firstColumn; ///< First column
  uint8_t lastColumn;  ///< Last column
  uint8_t firstLine;   ///< First line
  uint8_t lastLine;    ///< Last line
  uint8_t firstStrip;  ///< First strip
  uint8_t lastStrip;   ///< Last strip
};

std::ostream& operator<<(std::ostream& os, const PreCluster& data);

} // namespace mid
} // namespace o2

#endif /* O2_MID_PRECLUSTER_H */
