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

#ifndef O2_MCH_CONTOUR_POLYGON_H
#define O2_MCH_CONTOUR_POLYGON_H

#include <iostream>
#include <utility>
#include <vector>
#include <initializer_list>
#include <sstream>
#include <algorithm>
#include "BBox.h"
#include "Vertex.h"

namespace o2
{
namespace mch
{
namespace contour
{

template <typename T>
class Polygon;

template <typename T>
std::vector<o2::mch::contour::Vertex<T>> getVertices(const Polygon<T>& polygon);

template <typename T>
std::vector<o2::mch::contour::Vertex<T>> getSortedVertices(const Polygon<T>& polygon);

template <typename T>
class Polygon
{
 public:
  using size_type = typename std::vector<o2::mch::contour::Vertex<T>>::size_type;

  Polygon() = default;

  template <typename InputIterator>
  Polygon(InputIterator first, InputIterator last)
  {
    std::copy(first, last, std::back_inserter(mVertices));
  }

  Polygon(std::initializer_list<o2::mch::contour::Vertex<T>> args) : mVertices{args} {}

  o2::mch::contour::Vertex<T> firstVertex() const { return mVertices.front(); }

  size_type size() const { return mVertices.size(); }

  bool empty() const { return size() == 0; }

  o2::mch::contour::Vertex<T> operator[](int i) const { return mVertices[i]; }

  bool isCounterClockwiseOriented() const { return signedArea() > 0.0; }

  bool isManhattan() const
  {
    for (auto i = 0; i < mVertices.size() - 1; ++i) {
      if (!isVertical(mVertices[i], mVertices[i + 1]) && !isHorizontal(mVertices[i], mVertices[i + 1])) {
        return false;
      }
    }
    return true;
  }

  bool isClosed() const { return mVertices.back() == mVertices.front(); }

  bool contains(T x, T y) const;

  double signedArea() const
  {
    /// Compute the signed area of this polygon
    /// Algorithm from F. Feito, J.C. Torres and A. Urena,
    /// Comput. & Graphics, Vol. 19, pp. 595-600, 1995
    double area{0.0};
    for (auto i = 0; i < mVertices.size() - 1; ++i) {
      auto& current = mVertices[i];
      auto& next = mVertices[i + 1];
      area += current.x * next.y - next.x * current.y;
    }
    return area * 0.5;
  }

  void scale(T sx, T sy)
  {
    for (auto i = 0; i < mVertices.size(); ++i) {
      mVertices[i].x *= sx;
      mVertices[i].y *= sy;
    }
  }

  void translate(T dx, T dy)
  {
    for (auto i = 0; i < mVertices.size(); ++i) {
      mVertices[i].x += dx;
      mVertices[i].y += dy;
    }
  }

  friend std::ostream& operator<<(std::ostream& os, const Polygon<T>& polygon)
  {
    os << "POLYGON(";
    os << polygon.mVertices;
    os << ")";
    return os;
  }

 private:
  std::vector<o2::mch::contour::Vertex<T>> mVertices;
};

template <typename T>
Polygon<T> close(Polygon<T> polygon)
{
  if (!polygon.isClosed()) {
    auto vertices = getVertices(polygon);
    vertices.push_back(polygon.firstVertex());
    Polygon<T> pol(vertices.begin(), vertices.end());
    if (!pol.isManhattan()) {
      throw std::logic_error("closing resulted in non Manhattan polygon");
    }
    return pol;
  }
  return polygon;
}

template <typename T>
bool operator!=(const Polygon<T>& lhs, const Polygon<T>& rhs)
{
  return !(rhs == lhs);
}

/**
 * Two polygons are considered equal if they include the same set of vertices,
 * irrespective of orientation.
 */
template <typename T>
bool operator==(const Polygon<T>& lhs, const Polygon<T>& rhs)
{
  if (lhs.size() != rhs.size()) {
    return false;
  }

  auto l = getSortedVertices(lhs);
  auto r = getSortedVertices(rhs);

  if (l.size() != r.size()) {
    return false;
  }

  for (auto i = 0; i < l.size(); ++i) {
    if (l[i] != r[i]) {
      return false;
    }
  }
  return true;
}

template <typename T>
bool Polygon<T>::contains(T xp, T yp) const
{
  // Note that this algorithm yields unpredicatable result if the point xp,yp
  // is on one edge of the polygon. Should not generally matters, except when comparing
  // two different implementations maybe.
  //
  // TODO : look e.g. to http://alienryderflex.com/polygon/ for some possible optimizations
  // (e.g. pre-computation)
  //
  if (!isClosed()) {
    throw std::invalid_argument("contains method can only work with closed polygons");
  }

  auto j = mVertices.size() - 1;
  bool oddNodes{false};
  for (auto i = 0; i < mVertices.size(); i++) {
    if ((mVertices[i].y < yp && mVertices[j].y >= yp) || (mVertices[j].y < yp && mVertices[i].y >= yp)) {
      if (mVertices[i].x +
            (yp - mVertices[i].y) / (mVertices[j].y - mVertices[i].y) * (mVertices[j].x - mVertices[i].x) <
          xp) {
        oddNodes = !oddNodes;
      }
    }
    j = i;
  }
  return oddNodes;
}

template <typename T>
std::vector<o2::mch::contour::Vertex<T>> getVertices(const Polygon<T>& polygon)
{
  std::vector<o2::mch::contour::Vertex<T>> vertices;
  vertices.reserve(polygon.size());
  for (auto i = 0; i < polygon.size(); ++i) {
    vertices.push_back(polygon[i]);
  }
  return vertices;
}

template <typename T>
std::vector<o2::mch::contour::Vertex<T>> getSortedVertices(const Polygon<T>& polygon)
{
  std::vector<o2::mch::contour::Vertex<T>> vertices;
  auto size = polygon.size();
  if (polygon.isClosed()) {
    --size;
  }
  vertices.reserve(size);
  for (auto i = 0; i < size; ++i) {
    vertices.push_back(polygon[i]);
  }
  std::sort(vertices.begin(), vertices.end());
  return vertices;
}

template <typename T>
BBox<T> getBBox(const std::vector<Vertex<T>>& vertices)
{

  T xmin{std::numeric_limits<T>::max()};
  T xmax{std::numeric_limits<T>::lowest()};
  T ymin{std::numeric_limits<T>::max()};
  T ymax{std::numeric_limits<T>::lowest()};

  for (const auto& v : vertices) {
    xmin = std::min(xmin, v.x);
    xmax = std::max(xmax, v.x);
    ymin = std::min(ymin, v.y);
    ymax = std::max(ymax, v.y);
  }
  return {xmin, ymin, xmax, ymax};
}

template <typename T>
BBox<T> getBBox(const Polygon<T>& polygon)
{
  /// Return the bounding box (aka MBR, minimum bounding rectangle)
  /// of this polygon

  auto vertices = getVertices(polygon);
  return getBBox(vertices);
}

template <typename T>
BBox<T> getBBox(const std::vector<Polygon<T>>& polygons)
{
  /// Return the bounding box (aka MBR, minimum bounding rectangle)
  /// of this vector of polygons

  T xmin{std::numeric_limits<T>::max()};
  T xmax{std::numeric_limits<T>::lowest()};
  T ymin{std::numeric_limits<T>::max()};
  T ymax{std::numeric_limits<T>::lowest()};

  for (const auto& p : polygons) {
    auto b = getBBox(p);
    xmin = std::min(xmin, b.xmin());
    xmax = std::max(xmax, b.xmax());
    ymin = std::min(ymin, b.ymin());
    ymax = std::max(ymax, b.ymax());
  }
  return {xmin, ymin, xmax, ymax};
}

template <typename T>
auto squaredDistancePointToPolygon(const Vertex<T>& point, const Polygon<T>& polygon) -> decltype(point.x * point.x)
{
  T d{std::numeric_limits<T>::max()};
  for (auto i = 0; i < polygon.size() - 1; ++i) {
    auto s0 = polygon[i];
    auto s1 = polygon[i + 1];
    auto d2 = squaredDistanceOfPointToSegment(point, s0, s1);
    d = std::min(d2, d);
  }
  return d;
}

} // namespace contour
} // namespace mch
} // namespace o2

#endif
