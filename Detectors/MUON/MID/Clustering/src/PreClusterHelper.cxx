// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Clustering/src/PreClusterHelper.cxx
/// \brief  Implementation of the pre-clusters for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   16 April 2019
#include "MIDClustering/PreClusterHelper.h"

namespace o2
{
namespace mid
{

MpArea PreClusterHelper::getArea(const PreCluster& pc)
{
  /// Gets the area of the pre-cluster in the bending plane
  /// The method can also return the full area in the NBP
  /// However, in this case the area is always correct only in x.
  /// On the other hand, for the cut RPC the area is not well defined
  /// if the pre-cluster touches the column with the missing board.
  /// In this case we return the maximum rectangle (that includes the missing board)
  MpArea first = mMapping.stripByLocation(pc.firstStrip, pc.cathode, pc.firstLine, pc.firstColumn, pc.deId, false);
  float firstX = first.getXmin();
  float firstY = first.getYmin();
  float lastX = 0., lastY = 0.;
  if (pc.cathode == 0) {
    int nStripsInBetween = pc.lastStrip - pc.firstStrip + 16 * (pc.lastLine - pc.firstLine);
    lastX = first.getXmax();
    lastY = first.getYmax() + nStripsInBetween * mMapping.getStripSize(0, 0, pc.firstColumn, pc.deId);
  } else {
    MpArea last = mMapping.stripByLocation(pc.lastStrip, pc.cathode, 0, pc.lastColumn, pc.deId, false);
    lastX = last.getXmax();
    lastY = (last.getYmax() > first.getYmax()) ? last.getYmax() : first.getYmax();
    if (firstY > last.getYmin()) {
      firstY = last.getYmin();
    }
  }
  return MpArea{ firstX, firstY, lastX, lastY };
}

MpArea PreClusterHelper::getArea(int column, const PreCluster& pc)
{
  /// Gets the area of the pre-cluster in the non-bending plane in column
  if (column > pc.lastColumn || column < pc.firstColumn) {
    throw std::runtime_error("Required column is not in pre-cluster");
  }
  int firstStrip = (column > pc.firstColumn) ? 0 : pc.firstStrip;
  int lastStrip = (column < pc.lastColumn) ? mMapping.getNStripsNBP(column, pc.deId) - 1 : pc.lastStrip;

  MpArea first = mMapping.stripByLocation(firstStrip, 1, 0, column, pc.deId, false);
  MpArea last = mMapping.stripByLocation(lastStrip, 1, 0, column, pc.deId, false);
  return MpArea{ first.getXmin(), first.getYmin(), last.getXmax(), last.getYmax() };
}

} // namespace mid
} // namespace o2
