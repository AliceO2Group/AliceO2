#include "ITS3Reconstruction/TopologyDictionary.h"
#include "ITS3Base/SegmentationSuperAlpide.h"

namespace o2::its3 {


math_utils::Point3D<float> TopologyDictionary::getClusterCoordinates(int detID, const itsmft::CompCluster& cl) const {

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

math_utils::Point3D<float> TopologyDictionary::getClusterCoordinates(int detID, const itsmft::CompCluster& cl, const itsmft::ClusterPattern& patt, bool isGroup) {
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

}