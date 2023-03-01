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

math_utils::Point3D<float> TopologyDictionary::getClusterCoordinates(const its3::CompClusterExt& cl) const
{
  LOGP(debug, "Getting cluster coordinates from TopologyDictionaryITS3");
  static SegmentationSuperAlpide segmentations[6]{SegmentationSuperAlpide(0),
                                                  SegmentationSuperAlpide(0),
                                                  SegmentationSuperAlpide(1),
                                                  SegmentationSuperAlpide(1),
                                                  SegmentationSuperAlpide(2),
                                                  SegmentationSuperAlpide(2)}; // TODO: fix NLayers
  math_utils::Point3D<float> locCl;
  if (cl.getSensorID() >= 6) { // TODO: fix NLayers
    o2::itsmft::SegmentationAlpide::detectorToLocalUnchecked(cl.getRow(), cl.getCol(), locCl);
    locCl.SetX(locCl.X() + this->getXCOG(cl.getPatternID()));
    locCl.SetZ(locCl.Z() + this->getZCOG(cl.getPatternID()));
  } else {
    segmentations[cl.getSensorID()].detectorToLocalUnchecked(cl.getRow(), cl.getCol(), locCl);
    locCl.SetX(locCl.X() + this->getXCOG(cl.getPatternID()));
    locCl.SetZ(locCl.Z() + this->getZCOG(cl.getPatternID()));
    float xCurved{0.f}, yCurved{0.f};
    segmentations[cl.getSensorID()].flatToCurved(locCl.X(), locCl.Y(), xCurved, yCurved);
    locCl.SetXYZ(xCurved, yCurved, locCl.Z());
  }
  return locCl;
}

math_utils::Point3D<float> TopologyDictionary::getClusterCoordinates(const its3::CompClusterExt& cl, const itsmft::ClusterPattern& patt, bool isGroup)
{
  LOGP(debug, "Getting cluster coordinates from TopologyDictionaryITS3");
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
  if (cl.getSensorID() >= 6) { // TODO: fix NLayers
    o2::itsmft::SegmentationAlpide::detectorToLocalUnchecked(refRow + xCOG, refCol + zCOG, locCl);
  } else {
    segmentations[cl.getSensorID()].detectorToLocalUnchecked(refRow + xCOG, refCol + zCOG, locCl);
    float xCurved{0.f}, yCurved{0.f};
    segmentations[cl.getSensorID()].flatToCurved(locCl.X(), locCl.Y(), xCurved, yCurved);
    locCl.SetXYZ(xCurved, yCurved, locCl.Z());
  }
  return locCl;
}

} // namespace o2::its3