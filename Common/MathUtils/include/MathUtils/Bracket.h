// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Bracket.h
/// \brief Class to represent an interval and some operations over it
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_BRACKET_H
#define ALICEO2_BRACKET_H

#include <Rtypes.h>

namespace o2
{
namespace utils
{

template <typename T = float>
class Bracket
{
 public:
  enum Relation : int { Below = -1,
                        Inside = 0,
                        Above = 1 };

  Bracket(T minv, T maxv) : mMin(minv), mMax(maxv) {}
  Bracket(const Bracket<T>& src) = default;
  Bracket() = default;
  ~Bracket() = default;
  void set(T minv, T maxv)
  {
    mMin = minv;
    mMax = maxv;
  }
  bool operator<(const T rhs)
  {
    return mMax < rhs;
  }
  bool operator>(const T rhs)
  {
    return mMin > rhs;
  }

  bool operator<(const Bracket<T>& rhs)
  {
    return mMax < rhs.mMin;
  }
  bool operator>(const Bracket<T>& rhs)
  {
    return mMin > rhs.mMax;
  }
  bool operator==(const Bracket<T>& rhs)
  {
    return mMin == rhs.mMin && mMax == rhs.mMax;
  }

  void setMax(T v) { mMax = v; }
  void setMin(T v) { mMin = v; }
  T& max() { return mMax; }
  T& min() { return mMin; }
  T max() const { return mMax; }
  T min() const { return mMin; }
  T mean() const { return (mMin + mMax) / 2; }
  T delta() const { return mMax - mMin; }
  void update(T v)
  {
    // update limits
    if (v > mMax) {
      mMax = v;
    }
    if (v < mMin) {
      mMin = v;
    }
  }
  Relation isOutside(const Bracket<T>& t) const
  {
    ///< check if provided bracket is outside of this bracket
    return t.mMax < mMin ? Below : (t.mMin > mMax ? Above : Inside);
  }
  int isOutside(T t, T tErr) const
  {
    ///< check if the provided value t with error tErr is outside of the bracket
    return t + tErr < mMin ? Below : (t - tErr > mMax ? Above : Inside);
  }

 private:
  T mMin = 0, mMax = 0;

  ClassDefNV(Bracket, 1);
};
} // namespace utils
} // namespace o2

#endif
