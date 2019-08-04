// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @author  Laurent Aphecetche

#include "MCHMappingSegContour/SegmentationContours.h"
#include "MCHMappingSegContour/CathodeSegmentationContours.h"
#include "MCHContour/ContourCreator.h"
#include <iostream>

using namespace o2::mch::contour;

namespace o2
{
namespace mch
{
namespace mapping
{

BBox<double> getBBox(const Segmentation& seg) { return getBBox(getEnvelop(seg)); }

Contour<double> getEnvelop(const Segmentation& seg)
{
  std::vector<Polygon<double>> polygons;

  for (auto& contour : {getEnvelop(seg.nonBending()), getEnvelop(seg.bending())}) {
    for (auto& p : contour.getPolygons()) {
      polygons.push_back(p);
    }
  }
  return o2::mch::contour::createContour(polygons);
}

} // namespace mapping
} // namespace mch
} // namespace o2
