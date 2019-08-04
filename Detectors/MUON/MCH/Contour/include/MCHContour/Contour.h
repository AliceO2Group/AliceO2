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

#ifndef O2_MCH_CONTOUR_H
#define O2_MCH_CONTOUR_H

#include "Polygon.h"
#include <vector>
#include <cassert>
#include <initializer_list>

namespace o2
{
namespace mch
{
namespace contour
{

template <typename T>
std::vector<o2::mch::contour::Vertex<T>> getVertices(const std::vector<o2::mch::contour::Polygon<T>>& polygons)
{
  std::vector<o2::mch::contour::Vertex<T>> vertices;

  for (const auto& p : polygons) {
    auto pv = getVertices(p);
    vertices.insert(vertices.end(), pv.begin(), p.isClosed() ? pv.end() - 1 : pv.end());
  }

  return vertices;
}

template <typename T>
std::vector<o2::mch::contour::Vertex<T>> getSortedVertices(const std::vector<o2::mch::contour::Polygon<T>>& polygons)
{
  std::vector<Vertex<T>> vertices{getVertices(polygons)};
  std::sort(vertices.begin(), vertices.end());
  return vertices;
}

template <typename T>
bool isCounterClockwiseOriented(const std::vector<T>& polygons)
{
  for (const auto& p : polygons) {
    if (!p.isCounterClockwiseOriented()) {
      return false;
    }
  }
  return true;
}

template <typename T>
class Contour
{
 public:
  using size_type = typename std::vector<o2::mch::contour::Polygon<T>>::size_type;

  Contour() = default;

  Contour(std::initializer_list<o2::mch::contour::Polygon<T>> args) : mPolygons(args) {}

  size_type size() const { return mPolygons.size(); }

  o2::mch::contour::Polygon<T> operator[](int i) const { return mPolygons[i]; }

  bool empty() const { return size() == 0; }

  Contour<T>& addPolygon(const Polygon<T>& polygon)
  {
    mPolygons.push_back(polygon);
    return *this;
  }

  bool contains(T x, T y) const
  {
    for (const auto& p : mPolygons) {
      if (p.contains(x, y)) {
        return true;
      }
    }
    return false;
  }

  bool isClosed() const
  {
    for (const auto& p : mPolygons) {
      if (!p.isClosed()) {
        return false;
      }
    }
    return true;
  }

  bool isCounterClockwiseOriented() const { return o2::mch::contour::isCounterClockwiseOriented(mPolygons); }

  std::vector<o2::mch::contour::Vertex<T>> getVertices() const { return o2::mch::contour::getVertices(mPolygons); }

  std::vector<o2::mch::contour::Vertex<T>> getSortedVertices() const
  {
    return o2::mch::contour::getSortedVertices(mPolygons);
  }

  std::vector<o2::mch::contour::Polygon<T>> getPolygons() const { return mPolygons; }

  friend std::ostream& operator<<(std::ostream& os, const Contour<T>& contour)
  {
    os << contour.mPolygons;
    return os;
  }

 private:
  std::vector<o2::mch::contour::Polygon<T>> mPolygons;
};

template <typename T>
bool operator!=(const Contour<T>& lhs, const Contour<T>& rhs)
{
  return !(rhs == lhs);
}

/**
 * Two contours are considered equal if they contain
 * the same set of vertices
 */
template <typename T>
bool operator==(const Contour<T>& lhs, const Contour<T>& rhs)
{
  std::vector<Vertex<T>> vl{lhs.getSortedVertices()};
  std::vector<Vertex<T>> vr{rhs.getSortedVertices()};

  if (vl.size() != vr.size()) {
    return false;
  }
  for (auto i = 0; i < vl.size(); ++i) {
    if (vl[i] != vr[i]) {
      return false;
    }
  }
  return true;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<o2::mch::contour::Polygon<T>>& polygons)
{
  os << "MULTIPOLYGON(";

  for (auto j = 0; j < polygons.size(); ++j) {
    const Polygon<T>& p{polygons[j]};
    os << '(';
    for (auto i = 0; i < p.size(); ++i) {
      os << p[i].x << " " << p[i].y;
      if (i < p.size() - 1) {
        os << ',';
      }
    }
    os << ')';
    if (j < polygons.size() - 1) {
      os << ',';
    }
  }
  os << ')';
  return os;
}

template <typename T>
BBox<T> getBBox(const Contour<T>& contour)
{
  return getBBox(contour.getVertices());
}
} // namespace contour
} // namespace mch
} // namespace o2

#endif
