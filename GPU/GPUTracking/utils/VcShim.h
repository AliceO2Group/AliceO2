// Copyright 2020-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file VcShim.h
/// \brief Provides a basic fallback implementation for Vc
///
/// \author Felix Weiglhofer

#ifndef GPU_UTILS_VCSHIM_H
#define GPU_UTILS_VCSHIM_H

#ifndef GPUCA_NO_VC

#include <Vc/Vc> // IWYU pragma: export

#else

#include <algorithm>
#include <array>
#include <bitset>
#include <cstddef>
#include <type_traits>

namespace Vc
{

constexpr struct VectorSpecialInitializerZero {
} Zero;
constexpr struct AlignedTag {
} Aligned;

template <typename T>
typename T::vector_type& internal_data(T& v)
{
  return v.mData;
}

template <typename T>
const typename T::vector_type& internal_data(const T& v)
{
  return v.mData;
}

namespace Common
{

template <typename V, typename M>
class WriteMaskVector
{
 private:
  const M& mMask;
  V& mVec;

 public:
  using vector_type = V;
  using value_type = typename V::value_type;

  WriteMaskVector(V& v, const M& m) : mMask(m), mVec(v) {}

  WriteMaskVector& operator++(int)
  {
    for (size_t i = 0; i < mVec.size(); i++)
      mVec[i] += value_type(mMask[i]);
    return *this;
  }

  WriteMaskVector& operator=(const value_type& v)
  {
    for (size_t i = 0; i < mVec.size(); i++) {
      if (mMask[i])
        mVec[i] = v;
    }
    return *this;
  }

  WriteMaskVector& operator=(const vector_type& v)
  {
    for (size_t i = 0; i < mVec.size(); i++) {
      if (mMask[i])
        mVec[i] = v[i];
    }
    return *this;
  }
};

inline void prefetchMid(const void*) {}
inline void prefetchFar(const void*) {}
inline void prefetchForOneRead(const void*) {}

} // namespace Common

template <size_t N>
class fixed_size_simd_mask
{
 private:
  std::bitset<N> mData;

 public:
  bool isNotEmpty() const { return mData.any(); }

  typename std::bitset<N>::reference operator[](size_t i) { return mData[i]; }
  bool operator[](size_t i) const { return mData[i]; }

  fixed_size_simd_mask operator!() const
  {
    auto o = *this;
    o.mData.flip();
    return o;
  }

  fixed_size_simd_mask operator&&(const fixed_size_simd_mask& o) const
  {
    auto r = *this;
    r.mData &= o.mData;
    return r;
  }
};

template <typename T, size_t N>
class fixed_size_simd
{
 private:
  std::array<T, N> mData;

 public:
  using vector_type = std::array<T, N>;
  using value_type = T;
  using mask_type = fixed_size_simd_mask<N>;

  static constexpr size_t size() { return N; }

  fixed_size_simd() = default;
  explicit fixed_size_simd(VectorSpecialInitializerZero) { mData = {}; }

  template <typename U>
  fixed_size_simd(const fixed_size_simd<U, N>& w)
  {
    std::copy_n(internal_data(w).begin(), N, mData.begin());
  }

  fixed_size_simd(const T* d, AlignedTag) { std::copy_n(d, N, mData.begin()); }

  fixed_size_simd(const T& x) { mData.fill(x); }

  T& operator[](size_t i) { return mData[i]; }
  const T& operator[](size_t i) const { return mData[i]; }

  Common::WriteMaskVector<fixed_size_simd, mask_type> operator()(const mask_type& m) { return {*this, m}; }

  fixed_size_simd& operator=(const T& v)
  {
    for (auto& x : mData)
      x = v;
    return *this;
  }

  fixed_size_simd& operator+=(const T& v)
  {
    for (auto& x : mData)
      x += v;
    return *this;
  }

  template <typename U>
  fixed_size_simd& operator+=(const fixed_size_simd<U, N>& v)
  {
    for (size_t i = 0; i < N; i++)
      mData[i] += v[i];
    return *this;
  }

  fixed_size_simd& operator-=(const T& v)
  {
    for (auto& x : mData)
      x -= v;
    return *this;
  }

  template <typename U>
  fixed_size_simd& operator-=(const fixed_size_simd<U, N>& v)
  {
    for (size_t i = 0; i < N; i++)
      mData[i] -= v[i];
    return *this;
  }

  fixed_size_simd& operator*=(const T& v)
  {
    for (auto& x : mData)
      x *= v;
    return *this;
  }

  template <typename U>
  fixed_size_simd& operator*=(const fixed_size_simd<U, N>& v)
  {
    for (size_t i = 0; i < N; i++)
      mData[i] *= v[i];
    return *this;
  }

  fixed_size_simd& operator/=(const T& v)
  {
    for (auto& x : mData)
      x /= v;
    return *this;
  }

  template <typename U>
  fixed_size_simd& operator/=(const fixed_size_simd<U, N>& v)
  {
    for (size_t i = 0; i < N; i++)
      mData[i] /= v[i];
    return *this;
  }

  mask_type operator==(const T& v) const
  {
    mask_type m;
    for (size_t i = 0; i < N; i++)
      m[i] = mData[i] == v;
    return m;
  }

  mask_type operator!=(const T& v) const { return !(*this == v); }

  mask_type operator>(const T& v) const
  {
    mask_type m;
    for (size_t i = 0; i < N; i++)
      m[i] = mData[i] > v;
    return m;
  }

  mask_type operator>=(const T& v) const
  {
    mask_type m;
    for (size_t i = 0; i < N; i++)
      m[i] = mData[i] >= v;
    return m;
  }

  mask_type operator<(const T& v) const
  {
    mask_type m;
    for (size_t i = 0; i < N; i++)
      m[i] = mData[i] < v;
    return m;
  }

  friend vector_type& internal_data<>(fixed_size_simd& x);
  friend const vector_type& internal_data<>(const fixed_size_simd& x);
};

template <typename>
struct is_fixed_size_simd : std::false_type {
};

template <typename T, size_t N>
struct is_fixed_size_simd<fixed_size_simd<T, N>> : std::true_type {
};

template <typename S, typename T>
using EnableIfScalar = typename std::enable_if_t<
  !is_fixed_size_simd<typename std::decay_t<S>>::value && std::is_convertible_v<S, T>, int>;

template <typename T, typename U, size_t N>
fixed_size_simd<T, N> operator+(fixed_size_simd<T, N> a, const fixed_size_simd<U, N>& b)
{
  return a += b;
}

template <typename T, size_t N, typename S, EnableIfScalar<S, T> = 0>
fixed_size_simd<T, N> operator+(fixed_size_simd<T, N> a, const S& b)
{
  return a += static_cast<T>(b);
}

template <typename S, typename T, size_t N, EnableIfScalar<S, T> = 0>
fixed_size_simd<T, N> operator+(const S& a, fixed_size_simd<T, N> b)
{
  return b += static_cast<T>(a);
}

template <typename T, typename U, size_t N>
fixed_size_simd<T, N> operator-(fixed_size_simd<T, N> a, const fixed_size_simd<U, N>& b)
{
  return a -= b;
}

template <typename T, size_t N, typename S, EnableIfScalar<S, T> = 0>
fixed_size_simd<T, N> operator-(fixed_size_simd<T, N> a, const S& b)
{
  return a -= static_cast<T>(b);
}

template <typename S, typename T, size_t N, EnableIfScalar<S, T> = 0>
fixed_size_simd<T, N> operator-(const S& a, const fixed_size_simd<T, N>& b)
{
  fixed_size_simd<T, N> o;
  for (size_t i = 0; i < N; i++)
    o[i] = static_cast<T>(a) - b[i];
  return o;
}

template <typename T, typename U, size_t N>
fixed_size_simd<T, N> operator*(fixed_size_simd<T, N> a, const fixed_size_simd<U, N>& b)
{
  return a *= b;
}

template <typename T, size_t N, typename S, EnableIfScalar<S, T> = 0>
fixed_size_simd<T, N> operator*(fixed_size_simd<T, N> a, const S& b)
{
  return a *= static_cast<T>(b);
}

template <typename S, typename T, size_t N, EnableIfScalar<S, T> = 0>
fixed_size_simd<T, N> operator*(const S& a, fixed_size_simd<T, N> b)
{
  return b *= static_cast<T>(a);
}

template <typename T, typename U, size_t N>
fixed_size_simd<T, N> operator/(fixed_size_simd<T, N> a, const fixed_size_simd<U, N>& b)
{
  return a /= b;
}

template <typename T, size_t N, typename S, EnableIfScalar<S, T> = 0>
fixed_size_simd<T, N> operator/(fixed_size_simd<T, N> a, const S& b)
{
  return a /= static_cast<T>(b);
}

template <typename S, typename T, size_t N, EnableIfScalar<S, T> = 0>
fixed_size_simd<T, N> operator/(const S& a, const fixed_size_simd<T, N>& b)
{
  fixed_size_simd<T, N> o;
  for (size_t i = 0; i < N; i++)
    o[i] = static_cast<T>(a) / b[i];
  return o;
}

template <typename V>
V max(const V& a, const V& b)
{
  V o;
  for (size_t i = 0; i < a.size(); i++)
    o[i] = std::max(a[i], b[i]);
  return o;
}

} // namespace Vc

#endif // ifndef GPUCA_NO_VC

#endif
