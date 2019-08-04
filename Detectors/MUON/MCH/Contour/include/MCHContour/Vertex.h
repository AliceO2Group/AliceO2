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

#ifndef O2_MCH_CONTOUR_VERTEX_H
#define O2_MCH_CONTOUR_VERTEX_H

#include <iostream>
#include <iomanip>
#include "Helper.h"

namespace o2
{
namespace mch
{
namespace contour
{

template <typename T>
struct Vertex {
  T x;
  T y;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const Vertex<T>& vertex)
{
  os << '(' << vertex.x << ' ' << vertex.y << ')';
  return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<Vertex<T>>& vertices)
{
  for (auto i = 0; i < vertices.size(); ++i) {
    os << std::setw(5) << vertices[i].x << " " << std::setw(5) << vertices[i].y;
    if (i < vertices.size() - 1) {
      os << ',';
    }
  }
  os << ')';
  return os;
}

template <typename T>
bool operator<(const Vertex<T>& lhs, const Vertex<T>& rhs)
{
  if (lhs.y < rhs.y) {
    return true;
  }
  if (rhs.y < lhs.y) {
    return false;
  }
  return lhs.x < rhs.x;
}

template <typename T>
bool operator>(const Vertex<T>& lhs, const Vertex<T>& rhs)
{
  return rhs < lhs;
}

template <typename T>
bool operator<=(const Vertex<T>& lhs, const Vertex<T>& rhs)
{
  return !(rhs < lhs);
}

template <typename T>
bool operator>=(const Vertex<T>& lhs, const Vertex<T>& rhs)
{
  return !(lhs < rhs);
}

template <typename T>
bool isVertical(const Vertex<T>& a, const Vertex<T>& b)
{
  return impl::areEqual(a.x, b.x);
}

template <typename T>
bool isHorizontal(const Vertex<T>& a, const Vertex<T>& b)
{
  return impl::areEqual(a.y, b.y);
}

template <typename T>
Vertex<T> operator-(const Vertex<T>& a, const Vertex<T>& b)
{
  return {a.x - b.x, a.y - b.y};
}

template <typename T>
auto dot(const Vertex<T>& a, const Vertex<T>& b) -> decltype(a.x * b.x)
{
  // dot product
  return a.x * b.x + a.y * b.y;
}

template <typename T>
auto squaredDistance(const Vertex<T>& a, const Vertex<T>& b) -> decltype(a.x * b.x)
{
  return dot(a - b, a - b);
}

template <typename T>
auto squaredDistanceOfPointToSegment(const Vertex<T>& p, const Vertex<T>& p0, const Vertex<T>& p1)
  -> decltype(p0.x * p1.x)
{
  /// distance^2 of p to segment (p0,p1)
  auto v = p1 - p0;
  auto w = p - p0;

  auto c1 = dot(w, v);

  if (c1 <= 0) {
    return squaredDistance(p, p0);
  }

  auto c2 = dot(v, v);
  if (c2 <= c1) {
    return squaredDistance(p, p1);
  }

  auto b = c1 / c2;
  Vertex<T> pbase{p0.x + b * v.x, p0.y + b * v.y};
  return squaredDistance(p, pbase);
}

template <typename T>
bool operator==(const Vertex<T>& lhs, const Vertex<T>& rhs)
{
  return impl::areEqual(lhs.x, rhs.x) && impl::areEqual(lhs.y, rhs.y);
}

template <typename T>
bool operator!=(const Vertex<T>& lhs, const Vertex<T>& rhs)
{
  return !(lhs == rhs);
}

} // namespace contour
} // namespace mch
} // namespace o2

#endif
