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

#ifndef O2_MCH_CONTOUR_BBOX_H
#define O2_MCH_CONTOUR_BBOX_H

#include "Helper.h"
#include <stdexcept>
#include <ostream>

namespace o2
{
namespace mch
{
namespace contour
{

template <typename T>
class BBox
{
 public:
  BBox(T xmin, T ymin, T xmax, T ymax) : mXmin{xmin}, mXmax{xmax}, mYmin{ymin}, mYmax{ymax}
  {
    if (xmin >= xmax || ymin >= ymax) {
      throw std::invalid_argument("BBox should be created valid (xmin<xmax and ymin<ymax)");
    }
  }

  T xmin() const { return mXmin; }

  T xmax() const { return mXmax; }

  T ymin() const { return mYmin; }

  T ymax() const { return mYmax; }

  T xcenter() const { return (xmin() + xmax()) / T{2}; }

  T ycenter() const { return (ymin() + ymax()) / T{2}; }

  T width() const { return xmax() - xmin(); }

  T height() const { return ymax() - ymin(); }

  friend bool operator==(const BBox& lhs, const BBox& rhs)
  {
    return impl::areEqual(lhs.xmin(), rhs.xmin()) && impl::areEqual(lhs.ymin(), rhs.ymin()) &&
           impl::areEqual(lhs.xmax(), rhs.xmax()) && impl::areEqual(lhs.ymax(), rhs.ymax());
  }

  friend bool operator!=(const BBox& lhs, const BBox& rhs) { return !(rhs == lhs); }

  friend std::ostream& operator<<(std::ostream& os, const BBox& box)
  {
    os << "mTopLeft: " << box.xmin() << "," << box.ymax() << " mBottomRight: " << box.xmax() << "," << box.ymin();
    return os;
  }

  bool contains(const BBox<T>& a) const
  {
    return !(a.xmin() < xmin() || a.xmax() > xmax() || a.ymin() < ymin() || a.ymax() > ymax());
  }

 private:
  T mXmin;
  T mXmax;
  T mYmin;
  T mYmax;
};

template <typename T>
BBox<T> enlarge(const BBox<T>& box, T extraWidth, T extraHeight)
{
  return BBox<T>(box.xmin() - extraWidth / 2.0, box.ymin() - extraHeight / 2.0, box.xmax() + extraWidth / 2.0,
                 box.ymax() + extraHeight / 2.0);
}

template <typename T>
BBox<T> intersect(const BBox<T>& a, const BBox<T>& b)
{
  return BBox<T>{std::max(a.xmin(), b.xmin()), std::max(a.ymin(), b.ymin()), std::min(a.xmax(), b.xmax()),
                 std::min(a.ymax(), b.ymax())};
}

} // namespace contour
} // namespace mch
} // namespace o2

#endif
