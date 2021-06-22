// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TopologyDictionary.cxx

#include "ITS3Reconstruction/TopologyDictionary.h"
#include "ITS3Base/SegmentationSuperAlpide.h"

namespace o2::its3
{

math_utils::Point3D<float> TopologyDictionary::getClusterCoordinates(int detID, const its3::CompCluster& cl) const
{

  static SegmentationSuperAlpide segmentations[SegmentationSuperAlpide::NLayers]{SegmentationSuperAlpide(0), SegmentationSuperAlpide(1), SegmentationSuperAlpide(2), SegmentationSuperAlpide(3)};
  math_utils::Point3D<float> locCl;
  if (detID >= SegmentationSuperAlpide::NLayers) {

    o2::itsmft::SegmentationAlpide::detectorToLocalUnchecked(cl.getRow(), cl.getCol(), locCl);
  } else {
    segmentations[detID].detectorToLocalUnchecked(cl.getRow(), cl.getCol(), locCl);
  }
  locCl.SetX(locCl.X() + this->getXCOG(cl.getPatternID()));
  locCl.SetZ(locCl.Z() + this->getZCOG(cl.getPatternID()));
  return locCl;
}

math_utils::Point3D<float> TopologyDictionary::getClusterCoordinates(int detID, const its3::CompCluster& cl, const itsmft::ClusterPattern& patt, bool isGroup)
{
  static SegmentationSuperAlpide segmentations[SegmentationSuperAlpide::NLayers]{SegmentationSuperAlpide(0), SegmentationSuperAlpide(1), SegmentationSuperAlpide(2), SegmentationSuperAlpide(3)};

  auto refRow = cl.getRow();
  auto refCol = cl.getCol();
  float xCOG = 0, zCOG = 0;
  patt.getCOG(xCOG, zCOG);
  if (isGroup) {
    refRow -= round(xCOG);
    refCol -= round(zCOG);
  }
  math_utils::Point3D<float> locCl;
  if (detID >= SegmentationSuperAlpide::NLayers) {
    o2::itsmft::SegmentationAlpide::detectorToLocalUnchecked(refRow + xCOG, refCol + zCOG, locCl);
  } else {
    segmentations[detID].detectorToLocalUnchecked(refRow + xCOG, refCol + zCOG, locCl);
  }
  return locCl;
}

} // namespace o2::its3