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

#ifndef O2_MCH_CONTOUR_EDGE_H
#define O2_MCH_CONTOUR_EDGE_H

#include <utility>
#include <vector>
#include <iostream>
#include <iomanip>

#include "Interval.h"
#include "Vertex.h"

namespace o2
{
namespace mch
{
namespace contour
{
namespace impl
{

/**
 * A Manhattan edge is a segment of two vertices that
 * is guaranteed to be either horizontal or vertical
 *
 */
template <typename T>
class ManhattanEdge
{
 public:
  ManhattanEdge(Vertex<T> b = {}, Vertex<T> e = {});

  Vertex<T> begin() const { return mBegin; }

  Vertex<T> end() const { return mEnd; }

 private:
  Vertex<T> mBegin;
  Vertex<T> mEnd;
};

template <typename T>
bool isVertical(const ManhattanEdge<T>& edge)
{
  return isVertical(edge.begin(), edge.end());
}

template <typename T>
bool isHorizontal(const ManhattanEdge<T>& edge)
{
  return isHorizontal(edge.begin(), edge.end());
}

template <typename T>
ManhattanEdge<T>::ManhattanEdge(Vertex<T> b, Vertex<T> e) : mBegin(b), mEnd(e)
{
  if (!isVertical(*this) && !isHorizontal(*this)) {
    throw std::invalid_argument("edge should be either horizontal or vertical");
  }
}

template <typename T>
class VerticalEdge : public ManhattanEdge<T>
{
 public:
  VerticalEdge(T x = {}, T y1 = {}, T y2 = {}) : ManhattanEdge<T>({x, y1}, {x, y2}) {}
};

template <typename T>
class HorizontalEdge : public ManhattanEdge<T>
{
 public:
  HorizontalEdge(T y = {}, T x1 = {}, T x2 = {}) : ManhattanEdge<T>({x1, y}, {x2, y}) {}
};

template <typename T>
T top(const VerticalEdge<T>& edge)
{
  return std::max(edge.begin().y, edge.end().y);
}

template <typename T>
T bottom(const VerticalEdge<T>& edge)
{
  return std::min(edge.begin().y, edge.end().y);
}

template <typename T>
bool isLeftEdge(const VerticalEdge<T>& edge)
{
  return edge.begin().y > edge.end().y;
}

template <typename T>
bool isRightEdge(const VerticalEdge<T>& edge)
{
  return !isLeftEdge(edge);
}

template <typename T>
bool isTopToBottom(const VerticalEdge<T>& edge)
{
  return isLeftEdge(edge);
}

template <typename T>
bool isBottomToTop(const VerticalEdge<T>& edge)
{
  return !isTopToBottom(edge);
}

template <typename T>
T left(const HorizontalEdge<T>& edge)
{
  return std::min(edge.begin().x, edge.end().x);
}

template <typename T>
T right(const HorizontalEdge<T>& edge)
{
  return std::max(edge.begin().x, edge.end().x);
}

template <typename T>
bool isLeftToRight(const HorizontalEdge<T>& edge)
{
  return edge.begin().x < edge.end().x;
}

template <typename T>
bool isRightToLeft(const HorizontalEdge<T>& edge)
{
  return !isLeftToRight(edge);
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const ManhattanEdge<T>& edge)
{
  os << "[" << edge.begin() << "," << edge.end() << "]";
  return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const VerticalEdge<T>& edge)
{
  os << static_cast<const ManhattanEdge<T>>(edge);
  os << (isTopToBottom(edge) ? 'v' : '^');
  return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const HorizontalEdge<T>& edge)
{
  os << static_cast<const ManhattanEdge<T>>(edge);
  os << (isLeftToRight(edge) ? "->-" : "-<-");
  return os;
}

template <typename T>
bool operator==(const ManhattanEdge<T>& lhs, const ManhattanEdge<T>& rhs)
{
  return lhs.begin() == rhs.begin() && rhs.end() == rhs.end();
}

template <typename T>
bool operator!=(const ManhattanEdge<T>&& lhs, const ManhattanEdge<T>& rhs)
{
  return !(rhs == lhs);
}

} // namespace impl
} // namespace contour
} // namespace mch
} // namespace o2

#endif
