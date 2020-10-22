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
/// \author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch

#ifndef ALICEO2_BRACKET_H
#define ALICEO2_BRACKET_H

#include <GPUCommonRtypes.h>

namespace o2
{
namespace math_utils
{
namespace detail
{

template <typename T = float>
class Bracket
{
 public:
  enum Relation : int { Below = -1,
                        Inside = 0,
                        Above = 1 };

  Bracket(T minv, T maxv);
  Bracket() = default;
  Bracket(const Bracket<T>& src) = default;
  Bracket(Bracket<T>&& src) = default;
  Bracket& operator=(const Bracket<T>& src) = default;
  Bracket& operator=(Bracket<T>&& src) = default;
  ~Bracket() = default;

  bool operator<(T other) const;
  bool operator>(T other) const;

  bool operator<(const Bracket<T>& other) const;
  bool operator>(const Bracket<T>& other) const;
  bool operator==(const Bracket<T>& other) const;
  bool operator!=(const Bracket<T>& other) const;

  void setMax(T v) noexcept;
  void setMin(T v) noexcept;
  void set(T minv, T maxv) noexcept;

  T& getMax();
  T& getMin();

  T getMax() const;
  T getMin() const;
  T mean() const;
  T delta() const;
  bool isValid() const;
  bool isInvalid() const;
  void update(T v);
  Relation isOutside(const Bracket<T>& t) const;
  Relation isOutside(T t, T tErr) const;

 private:
  T mMin{};
  T mMax{};

  ClassDefNV(Bracket, 2);
};

template <typename T>
Bracket<T>::Bracket(T minv, T maxv) : mMin(minv), mMax(maxv)
{
}

template <typename T>
inline bool Bracket<T>::operator<(T other) const
{
  return mMax < other;
}

template <typename T>
inline bool Bracket<T>::operator>(T other) const
{
  return mMin > other;
}

template <typename T>
inline bool Bracket<T>::operator<(const Bracket<T>& other) const
{
  return *this < other.mMin;
}

template <typename T>
inline bool Bracket<T>::operator>(const Bracket<T>& other) const
{
  return *this > other.mMax;
}

template <typename T>
inline bool Bracket<T>::operator==(const Bracket<T>& rhs) const
{
  return mMin == rhs.mMin && mMax == rhs.mMax;
}

template <typename T>
inline bool Bracket<T>::operator!=(const Bracket<T>& rhs) const
{
  return !(*this == rhs);
}

template <typename T>
inline void Bracket<T>::setMax(T v) noexcept
{
  mMax = v;
}

template <typename T>
inline void Bracket<T>::setMin(T v) noexcept
{
  mMin = v;
}

template <typename T>
inline void Bracket<T>::set(T minv, T maxv) noexcept
{
  this->setMin(minv);
  this->setMax(maxv);
}

template <typename T>
inline T& Bracket<T>::getMax()
{
  return mMax;
}

template <typename T>
inline T& Bracket<T>::getMin()
{
  return mMin;
}

template <typename T>
inline T Bracket<T>::getMax() const
{
  return mMax;
}

template <typename T>
inline T Bracket<T>::getMin() const
{
  return mMin;
}

template <typename T>
inline T Bracket<T>::mean() const
{
  return (mMin + mMax) / 2;
}
template <typename T>
inline T Bracket<T>::delta() const
{
  return mMax - mMin;
}
template <typename T>
inline bool Bracket<T>::isValid() const
{
  return mMax >= mMin;
}
template <typename T>
inline bool Bracket<T>::isInvalid() const
{
  return mMin > mMax;
}
template <typename T>
inline void Bracket<T>::update(T v)
{
  // update limits
  if (v > mMax) {
    mMax = v;
  }
  if (v < mMin) {
    mMin = v;
  }
}

template <typename T>
inline typename Bracket<T>::Relation Bracket<T>::isOutside(const Bracket<T>& t) const
{
  ///< check if provided bracket is outside of this bracket
  return t.mMax < mMin ? Below : (t.mMin > mMax ? Above : Inside);
}
template <typename T>
inline typename Bracket<T>::Relation Bracket<T>::isOutside(T t, T tErr) const
{
  ///< check if the provided value t with error tErr is outside of the bracket
  return t + tErr < mMin ? Below : (t - tErr > mMax ? Above : Inside);
}

} // namespace detail
} // namespace math_utils
} // namespace o2

#endif
