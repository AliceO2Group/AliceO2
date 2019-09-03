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

#ifndef O2_MCH_CONTOUR_CONTOURCREATOR_H
#define O2_MCH_CONTOUR_CONTOURCREATOR_H

#include "ContourCreator.inl"

namespace o2
{
namespace mch
{
namespace contour
{

/// Merge polygons into a contour
///
/// based on (one of the) algorithms in
/// Diane L. Souvaine and Iliana Bjorling-Sachs,
/// Proceedings of the IEEE, Vol. 80, No. 9, September 1992, p. 1449
///
template <typename T>
Contour<T> createContour(const std::vector<Polygon<T>>& polygons)
{
  if (polygons.empty()) {
    return {};
  }

  if (!isCounterClockwiseOriented(polygons)) {
    throw std::invalid_argument("polygons should be oriented counterclockwise");
  }

  // trivial case : only one input polygon
  if (polygons.size() == 1) {
    Contour<T> trivialContour;
    trivialContour.addPolygon(polygons.front());
    return trivialContour;
  }

  std::vector<impl::VerticalEdge<T>> polygonVerticalEdges{impl::getVerticalEdges(polygons)};

  sortVerticalEdges(polygonVerticalEdges);

  // Initialize the segment tree that is used by the sweep() function
  std::unique_ptr<impl::Node<T>> segmentTree{impl::createSegmentTree(impl::getYPositions(polygons))};

  // Find the vertical edges of the merged contour. This is the meat of the algorithm...
  std::vector<impl::VerticalEdge<T>> contourVerticalEdges{impl::sweep(segmentTree.get(), polygonVerticalEdges)};

  // Deduce the horizontal edges from the vertical ones
  std::vector<impl::HorizontalEdge<T>> contourHorizontalEdges{impl::verticalsToHorizontals(contourVerticalEdges)};

  return impl::finalizeContour(contourVerticalEdges, contourHorizontalEdges);
}

template <typename T>
Contour<T> getEnvelop(const std::vector<Contour<T>>& list)
{
  /// get the envelop of a collection of contours
  std::vector<o2::mch::contour::Polygon<T>> polygons;
  for (const auto& c : list) {
    for (auto j = 0; j < c.size(); ++j) {
      polygons.push_back(c[j]);
    }
  }
  return createContour(polygons);
}

} // namespace contour
} // namespace mch
} // namespace o2

#endif
