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

/// \file TopologyDictionary.cxx

#include "ITS3Reconstruction/TopologyDictionary.h"
#include "ITS3Base/SegmentationSuperAlpide.h"
#include "ITSMFTBase/SegmentationAlpide.h"

namespace o2::its3
{

math_utils::Point3D<float> TopologyDictionary::getClusterCoordinates(int detID, const its3::CompCluster& cl) const
{

  static SegmentationSuperAlpide segmentations[6]{SegmentationSuperAlpide(0),
                                                  SegmentationSuperAlpide(0),
                                                  SegmentationSuperAlpide(1),
                                                  SegmentationSuperAlpide(1),
                                                  SegmentationSuperAlpide(2),
                                                  SegmentationSuperAlpide(2)}; // TODO: fix NLayers
  math_utils::Point3D<float> locCl;
  if (detID >= 6) { // TODO: fix NLayers
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
  static SegmentationSuperAlpide segmentations[6]{SegmentationSuperAlpide(0),
                                                  SegmentationSuperAlpide(0),
                                                  SegmentationSuperAlpide(1),
                                                  SegmentationSuperAlpide(1),
                                                  SegmentationSuperAlpide(2),
                                                  SegmentationSuperAlpide(2)}; // TODO: fix NLayers

  auto refRow = cl.getRow();
  auto refCol = cl.getCol();
  float xCOG = 0, zCOG = 0;
  patt.getCOG(xCOG, zCOG);
  if (isGroup) {
    refRow -= round(xCOG);
    refCol -= round(zCOG);
  }
  math_utils::Point3D<float> locCl;
  if (detID >= 6) { // TODO: fix NLayers
    o2::itsmft::SegmentationAlpide::detectorToLocalUnchecked(refRow + xCOG, refCol + zCOG, locCl);
  } else {
    segmentations[detID].detectorToLocalUnchecked(refRow + xCOG, refCol + zCOG, locCl);
  }
  return locCl;
}

} // namespace o2::its3