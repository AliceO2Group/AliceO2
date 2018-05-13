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
#include "MCHContour/ContourCreator.h"

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

  for (auto& contour : getDualSampaContours(seg)) {
    for (auto& p : contour.getPolygons()) {
      polygons.push_back(p);
    }
  }

  return o2::mch::contour::createContour(polygons);
}

std::vector<std::vector<int>> getPadChannels(const Segmentation& seg)
{
  std::vector<std::vector<int>> dualSampaPads;

  for (auto i = 0; i < seg.nofDualSampas(); ++i) {
    std::vector<int> pads;
    seg.forEachPadInDualSampa(seg.dualSampaId(i), [&pads, &seg](int paduid) {
      double x = seg.padPositionX(paduid);
      double y = seg.padPositionY(paduid);
      double dx = seg.padSizeX(paduid) / 2.0;
      double dy = seg.padSizeY(paduid) / 2.0;

      pads.emplace_back(seg.padDualSampaChannel(paduid));
    });
    dualSampaPads.push_back(pads);
  }

  return dualSampaPads;
}

std::vector<Polygon<double>> getPadPolygons(const Segmentation& seg, int dualSampaId)
{
  std::vector<Polygon<double>> pads;
  seg.forEachPadInDualSampa(dualSampaId, [&pads, &seg](int paduid) {
    double x = seg.padPositionX(paduid);
    double y = seg.padPositionY(paduid);
    double dx = seg.padSizeX(paduid) / 2.0;
    double dy = seg.padSizeY(paduid) / 2.0;

    pads.emplace_back(Polygon<double>{
      { x - dx, y - dy }, { x + dx, y - dy }, { x + dx, y + dy }, { x - dx, y + dy }, { x - dx, y - dy } });
  });
  return pads;
}

std::vector<std::vector<Polygon<double>>> getPadPolygons(const Segmentation& seg)
{
  std::vector<std::vector<Polygon<double>>> dualSampaPads;

  for (auto i = 0; i < seg.nofDualSampas(); ++i) {
    dualSampaPads.push_back(getPadPolygons(seg, seg.dualSampaId(i)));
  }

  return dualSampaPads;
}

o2::mch::contour::Contour<double> getDualSampaContour(const Segmentation& seg, int dualSampaId)
{
  auto padPolygons = getPadPolygons(seg, dualSampaId);
  return o2::mch::contour::createContour(padPolygons);
}

std::vector<o2::mch::contour::Contour<double>> getDualSampaContours(const Segmentation& seg)
{
  std::vector<o2::mch::contour::Contour<double>> contours;
  for (auto i = 0; i < seg.nofDualSampas(); ++i) {
    contours.push_back(o2::mch::contour::createContour(getPadPolygons(seg, seg.dualSampaId(i))));
  }
  return contours;
}

} // namespace mapping
} // namespace mch
} // namespace o2
