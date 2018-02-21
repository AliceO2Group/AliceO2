//
// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @author  Laurent Aphecetche

#ifndef O2_MCH_CONTOUR_CONTOURCREATOR_INL
#define O2_MCH_CONTOUR_CONTOURCREATOR_INL

#include <utility>
#include <vector>
#include <iostream>
#include "Edge.h"
#include "Polygon.h"
#include "Contour.h"
#include "SegmentTree.h"
#include <algorithm>

namespace o2 {
namespace mch {
namespace contour {
namespace impl {

template<typename T>
void sortVerticalEdges(std::vector<VerticalEdge<T>> &edges)
{
// sort vertical edges in ascending x order
// if same x, insure that left edges are before right edges
// within same x, order by increasing bottommost y
// Mind your steps ! This sorting is critical to the contour merging algorithm !

  std::sort(edges.begin(), edges.end(),
            [](const VerticalEdge<T> &e1, const VerticalEdge<T> &e2) {

              auto x1 = e1.begin().x;
              auto x2 = e2.begin().x;

              auto y1 = bottom(e1);
              auto y2 = bottom(e2);

              if (areEqual(x1, x2)) {
                if (isLeftEdge(e1) && isRightEdge(e2)) {
                  return true;
                }
                if (isRightEdge(e1) && isLeftEdge(e2)) {
                  return false;
                }
                return y1 < y2;
              } else if (x1 < x2) {
                return true;
              } else {
                return false;
              }
            });
}

template<typename T>
Interval<T> interval(const VerticalEdge<T> &edge)
{
  auto y1 = edge.begin().y;
  auto y2 = edge.end().y;
  return y2 > y1 ? Interval<T>(y1, y2) : Interval<T>(y2, y1);
}

template<typename T>
std::vector<VerticalEdge<T>> getVerticalEdges(const Polygon<T> &polygon)
{
  /// Return the vertical edges of the input polygon
  std::vector<VerticalEdge<T>> edges;
  for (auto i = 0; i < polygon.size() - 1; ++i) {
    auto current = polygon[i];
    auto next = polygon[i + 1];
    if (current.x == next.x) {
      edges.push_back({current.x, current.y, next.y});
    }
  }
  return edges;
}

template<typename T>
std::vector<VerticalEdge<T>> getVerticalEdges(const std::vector<Polygon<T>> &polygons)
{
  std::vector<VerticalEdge<T>> edges;
  for (const auto &p:polygons) {
    auto e = getVerticalEdges(p);
    edges.insert(edges.end(), e.begin(), e.end());
  }
  return edges;
}

template<typename T>
T getX(const Vertex<T> &v)
{
  return v.x;
}

template<typename T>
T getY(const Vertex<T> &v)
{
  return v.y;
}

template<typename T>
using GetVertexPosFunc = T(*)(const Vertex<T> &);

template<typename T>
std::vector<T> getPositions(const std::vector<Polygon<T>> &polygons, GetVertexPosFunc<T> func)
{
  std::vector<T> ypos;
  for (auto i = 0; i < polygons.size(); ++i) {
    for (auto j = 0; j < polygons[i].size(); ++j) {
      ypos.push_back(func(polygons[i][j]));
    }
  }
  std::sort(ypos.begin(), ypos.end());
  auto last = std::unique(ypos.begin(), ypos.end(),
                          [](const T &a, const T &b) { return areEqual(a, b); });
  ypos.erase(last, ypos.end());
  return ypos;
}

template<typename T>
std::vector<T> getYPositions(const std::vector<Polygon<T>> &polygons)
{
  return getPositions(polygons, getY);
}

template<typename T>
std::vector<T> getXPositions(const std::vector<Polygon<T>> &polygons)
{
  return getPositions(polygons, getX);
}

template<typename T>
std::vector<VerticalEdge<T>> sweep(Node<T> *segmentTree, const std::vector<VerticalEdge<T>> &polygonVerticalEdges)
{
  std::vector<VerticalEdge<T>> contourVerticalEdges;

  std::vector<Interval<T>> edgeStack;

  for (auto i = 0; i < polygonVerticalEdges.size(); ++i) {

    const auto &edge = polygonVerticalEdges[i];
    auto ival = interval(edge);

    if (isLeftEdge(edge)) {
      segmentTree->contribution(ival, edgeStack);
      segmentTree->insertInterval(ival);
    } else {
      segmentTree->deleteInterval(ival);
      segmentTree->contribution(ival, edgeStack);
    }

    auto e1{edge};

    if (i < polygonVerticalEdges.size() - 1) {
      e1 = polygonVerticalEdges[i + 1];
    }

    if ((isLeftEdge(edge) != isLeftEdge(e1)) ||
        (!areEqual(edge.begin().x, e1.begin().x)) ||
        (i == polygonVerticalEdges.size() - 1)) {
      for (auto es :edgeStack) {
        contourVerticalEdges.push_back(isRightEdge(edge)
                                       ? VerticalEdge<T>{edge.begin().x, es.begin(), es.end()} :
                                       VerticalEdge<T>{edge.begin().x, es.end(), es.begin()});
      }
      edgeStack.clear();
    }
  }

  return contourVerticalEdges;
}

/**
 * Generates horizontal edges from the vertical ones
 * The horizontals are ordered relative to the verticals, i.e. the first horizontal
 * should be the edge __following__ the first vertical, etc...
 *
 * @param verticals
 * @return the horizontals, in the relevant order
 */
template<typename T>
std::vector<HorizontalEdge<T>> verticalsToHorizontals(const std::vector<VerticalEdge<T>> &verticals)
{
  std::vector<HorizontalEdge<T>> horizontals(verticals.size());

  using VertexWithRef = std::pair<Vertex<T>, T>;
  std::vector<VertexWithRef> vertices;

  for (auto i = 0; i < verticals.size(); ++i) {
    const auto &edge = verticals[i];
    vertices.push_back({edge.begin(), i});
    vertices.push_back({edge.end(), i});
  }

  std::sort(vertices.begin(), vertices.end(),
            [](const VertexWithRef &v1, const VertexWithRef &v2) {
              return v1.first < v2.first;
            });

  for (auto i = 0; i < vertices.size() / 2; ++i) {
    const auto &p1 = vertices[2 * i];
    const auto &p2 = vertices[2 * i + 1];
    const VerticalEdge<T> &refEdge = verticals[p1.second];
    auto e = p1.first.x;
    auto b = p2.first.x;
    if ((areEqual(p1.first.y, bottom(refEdge)) &&
         isLeftEdge(refEdge)) ||
        (areEqual(p1.first.y, top(refEdge)) &&
         isRightEdge(refEdge))) {
      std::swap(b, e);
    }
    HorizontalEdge<T> h{p1.first.y, b, e};
// which vertical edge is preceding this horizontal ?
    int preceding = p1.second;
    int next = p2.second;
    if (b > e) {
      std::swap(preceding, next);
    }
    horizontals[preceding] = h;
  }
  return horizontals;
}

template<typename T>
Contour<T> finalizeContour(const std::vector<VerticalEdge<T>> &verticals,
                           const std::vector<HorizontalEdge<T>> &horizontals)
{
  if (verticals.size() != horizontals.size()) {
    throw std::invalid_argument("should get the same number of verticals and horizontals");
  }

  for (auto i = 0; i < verticals.size(); ++i) {
    if (horizontals[i].begin() != verticals[i].end()) {
      throw std::invalid_argument("got an horizontal edge not connected to its (supposedly) preceding vertical edge");
    }
  }

  std::vector<ManhattanEdge<T>> all;

  for (auto i = 0; i < verticals.size(); ++i) {
    all.push_back(verticals[i]);
    all.push_back(horizontals[i]);
  }

  Contour<T> contour;

  std::vector<bool> alreadyAdded(all.size(), false);
  std::vector<int> inorder;

  int nofUsed{0};
  int iCurrent{0};

  ManhattanEdge<T> startSegment{all[iCurrent]};

  while (nofUsed < all.size()) {

    const ManhattanEdge<T> &currentSegment{all[iCurrent]};
    inorder.push_back(iCurrent);
    alreadyAdded[iCurrent] = true;
    ++nofUsed;
    if (currentSegment.end() == startSegment.begin()) {
      if (inorder.empty()) {
        throw std::runtime_error("got an empty polygon");
      }
      std::vector<Vertex<T>> vertices;
      vertices.reserve(inorder.size());
      for (auto i:inorder) { vertices.push_back(all[i].begin()); }
      Polygon<T> polygon(vertices.begin(), vertices.end());
      contour.addPolygon(close(polygon));
      iCurrent = std::distance(alreadyAdded.begin(), std::find_if(alreadyAdded.begin(), alreadyAdded.end(),
                                                                  [](bool a) { return a == false; }));
      startSegment = all[iCurrent];
      inorder.clear();
    }

    for (auto i = 0; i < alreadyAdded.size(); ++i) {
      if (i != iCurrent && alreadyAdded[i] == false) {
        if (currentSegment.end() == all[i].begin()) {
          iCurrent = i;
          break;
        }
      }
    }
  }

  return contour;
}


}
}
}
}

#endif //ALO_CONTOURCREATOR_INL
