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


#ifndef O2_MCH_CONTOUR_INTERVAL_H
#define O2_MCH_CONTOUR_INTERVAL_H

#include <vector>
#include <ostream>
#include "Helper.h"

namespace o2 {
namespace mch {
namespace contour {
namespace impl {
template<typename T>
class Interval
{
  public:

    Interval(T b = {}, T e = {}) : mBegin(b), mEnd(e)
    {
      if (b > e || areEqual(b, e)) {
        throw std::invalid_argument("begin should be strictly < end");
      }
    }

    bool isFullyContainedIn(Interval i) const
    {
      return (i.begin() < mBegin || areEqual(i.begin(), mBegin)) &&
             (mEnd < i.end() || areEqual(i.end(), mEnd));
    }

    T begin() const
    { return mBegin; }

    T end() const
    { return mEnd; }

    bool extend(const Interval &i)
    {
      if (areEqual(i.begin(), end())) {
        mEnd = i.end();
        return true;
      } else {
        return false;
      }
    }

    bool operator==(const Interval &rhs) const
    {
      return areEqual(mBegin, rhs.mBegin) &&
             areEqual(mEnd, rhs.mEnd);
    }

    bool operator!=(const Interval &rhs) const
    {
      return !(rhs == *this);
    }

  private:
    T mBegin;
    T mEnd;
};

template<typename T>
inline std::ostream &operator<<(std::ostream &os, const Interval<T> &i)
{
  os << "[" << i.begin() << "," << i.end() << "]";
  return os;
}

}
}
}
}

#endif
