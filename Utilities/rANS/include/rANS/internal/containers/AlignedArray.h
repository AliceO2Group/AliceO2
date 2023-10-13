// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AlignedArray.h
/// @author Michael Lettrich
/// @brief  Memory aligned array used for SIMD operations

#ifndef RANS_INTERNAL_CONTAINERS_ALIGNEDARRAY_H_
#define RANS_INTERNAL_CONTAINERS_ALIGNEDARRAY_H_

#include <cstring>
#include <cassert>
#include <cstdint>
#include <type_traits>

#include <fmt/format.h>
#include <gsl/span>

#include "rANS/internal/common/utils.h"
#include "rANS/internal/common/simdtypes.h"

namespace o2::rans::internal::simd
{

template <typename array_T>
class AlignedArrayIterator
{
  template <typename T>
  struct getValueType {
    using type = std::conditional_t<std::is_const_v<T>, const typename T::value_type, typename T::value_type>;
  };

 public:
  using difference_type = std::ptrdiff_t;
  using value_type = gsl::span<typename getValueType<array_T>::type, array_T::nElementsPerLane()>;
  using pointer = value_type*;
  using reference = value_type&;
  using iterator_category = std::random_access_iterator_tag;

  inline constexpr AlignedArrayIterator() noexcept = default;

  inline constexpr AlignedArrayIterator(array_T* array, difference_type index) noexcept : mIndex{index} {};
  inline constexpr AlignedArrayIterator(const AlignedArrayIterator& iter) noexcept = default;
  inline constexpr AlignedArrayIterator(AlignedArrayIterator&& iter) noexcept = default;
  inline constexpr AlignedArrayIterator& operator=(const AlignedArrayIterator& other) noexcept = default;
  inline constexpr AlignedArrayIterator& operator=(AlignedArrayIterator&& other) noexcept = default;
  inline ~AlignedArrayIterator() noexcept = default;

  // pointer arithmetics
  inline constexpr AlignedArrayIterator& operator++() noexcept
  {
    ++mIndex;
    return *this;
  };

  inline constexpr AlignedArrayIterator operator++(int) noexcept
  {
    auto res = *this;
    ++(*this);
    return res;
  };

  inline constexpr AlignedArrayIterator& operator--() noexcept
  {
    --mIndex;
    return *this;
  };

  inline constexpr AlignedArrayIterator operator--(int) noexcept
  {
    auto res = *this;
    --(*this);
    return res;
  };

  inline constexpr AlignedArrayIterator& operator+=(difference_type i) noexcept
  {
    mIndex += i;
    return *this;
  };

  inline constexpr AlignedArrayIterator operator+(difference_type i) const noexcept
  {
    auto tmp = *const_cast<AlignedArrayIterator*>(this);
    return tmp += i;
  }

  inline constexpr AlignedArrayIterator& operator-=(difference_type i) noexcept
  {
    mIndex -= i;
    return *this;
  };

  inline constexpr AlignedArrayIterator operator-(difference_type i) const noexcept
  {
    auto tmp = *const_cast<AlignedArrayIterator*>(this);
    return tmp -= i;
  };

  inline constexpr difference_type operator-(const AlignedArrayIterator& other) const noexcept
  {
    return this->mIter - other.mIter;
  };

  // comparison
  inline constexpr bool operator==(const AlignedArrayIterator& other) const noexcept { return this->mIndex == other.mIndex; };
  inline constexpr bool operator!=(const AlignedArrayIterator& other) const noexcept { return this->mIndex != other.mIndex; };
  inline constexpr bool operator<(const AlignedArrayIterator& other) const noexcept { return this->mIndex < other->mIndex; };
  inline constexpr bool operator>(const AlignedArrayIterator& other) const noexcept { return this->mIndex > other->mIndex; };
  inline constexpr bool operator>=(const AlignedArrayIterator& other) const noexcept { return this->mIndex >= other->mIndex; };
  inline constexpr bool operator<=(const AlignedArrayIterator& other) const noexcept { return this->mIndex <= other->mIndex; };

  // dereference
  inline constexpr value_type operator*() const noexcept { return (*mContainer)[mIndex]; };

  inline constexpr value_type operator[](difference_type i) const noexcept { return (*mContainer)[mIndex + i]; };

 private:
  array_T* mContainer;
  difference_type mIndex{};
};

template <typename T, SIMDWidth width_V, size_t size_V = 1>
class alignas(getAlignment(width_V)) AlignedArray
{
 public:
  using value_type = T;
  using pointer = value_type*;
  using reference = value_type&;
  using iterator = AlignedArrayIterator<AlignedArray>;
  using const_iterator = AlignedArrayIterator<const AlignedArray>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  static inline constexpr size_t size() noexcept { return size_V; };
  static inline constexpr size_t nElementsPerLane() noexcept { return getElementCount<value_type>(width_V); };
  static inline constexpr size_t nElements() noexcept { return size() * nElementsPerLane(); };

  inline constexpr AlignedArray() noexcept {}; // NOLINT

  template <typename elem_T, std::enable_if_t<std::is_convertible_v<elem_T, value_type>, bool> = true>
  inline constexpr AlignedArray(elem_T value) noexcept
  {
    for (auto& elem : mData) {
      elem = static_cast<value_type>(value);
    }
  };

  template <typename... Args, std::enable_if_t<(sizeof...(Args) == AlignedArray<T, width_V, size_V>::nElements()) && std::is_convertible_v<std::common_type_t<Args...>, value_type>, bool> = true>
  inline constexpr AlignedArray(Args... args) noexcept : mData{static_cast<value_type>(args)...} {};

  inline constexpr const T* data() const noexcept { return mData; };
  inline constexpr T* data() noexcept { return const_cast<T*>(static_cast<const AlignedArray&>(*this).data()); };
  inline constexpr const_iterator begin() const noexcept { return {this, 0}; };
  inline constexpr const_iterator end() const noexcept { return {this, size()}; };
  inline constexpr iterator begin() noexcept { return {this, 0}; };
  inline constexpr iterator end() noexcept { return {this, size()}; };
  inline constexpr const_reverse_iterator rbegin() const noexcept { return std::reverse_iterator(this->end()); };
  inline constexpr const_reverse_iterator rend() const noexcept { return std::reverse_iterator(this->begin()); };
  inline constexpr reverse_iterator rbegin() noexcept { return std::reverse_iterator(this->end()); };
  inline constexpr reverse_iterator rend() noexcept { return std::reverse_iterator(this->begin()); };

  inline constexpr gsl::span<T, nElementsPerLane()> operator[](size_t idx) { return gsl::span<T, nElementsPerLane()>{mData + idx * nElementsPerLane(), nElementsPerLane()}; };

  inline constexpr gsl::span<const T, nElementsPerLane()> operator[](size_t idx) const { return gsl::span<const T, nElementsPerLane()>{mData + idx * nElementsPerLane(), nElementsPerLane()}; };

  inline constexpr const T& operator()(size_t idx, size_t elem) const
  {
    return (*this)[idx][elem];
  };

  inline constexpr T& operator()(size_t idx, size_t elem) { return const_cast<T&>(static_cast<const AlignedArray&>(*this)(idx, elem)); };

  inline constexpr const T& operator()(size_t idx) const
  {
    return *(this->data() + idx);
  };

  inline constexpr T& operator()(size_t idx) { return const_cast<T&>(static_cast<const AlignedArray&>(*this)(idx)); };

 private:
  T mData[nElements()]{};
};

template <SIMDWidth width_V, size_t size_V = 1>
using pd_t = AlignedArray<double_t, width_V, size_V>;
template <SIMDWidth width_V, size_t size_V = 1>
using epi64_t = AlignedArray<uint64_t, width_V, size_V>;
template <SIMDWidth width_V, size_t size_V = 1>
using epi32_t = AlignedArray<uint32_t, width_V, size_V>;
template <SIMDWidth width_V, size_t size_V = 1>
using epi16_t = AlignedArray<uint16_t, width_V, size_V>;
template <SIMDWidth width_V, size_t size_V = 1>
using epi8_t = AlignedArray<uint8_t, width_V, size_V>;

template <typename T>
struct simdWidth;

template <typename T, SIMDWidth simd_V>
struct simdWidth<AlignedArray<T, simd_V>> : public std::integral_constant<SIMDWidth, simd_V> {
};

template <typename T>
inline constexpr SIMDWidth simdWidth_v = simdWidth<T>::value;

template <typename T>
struct elementCount;

template <typename T, SIMDWidth simd_V, size_t size_V>
struct elementCount<AlignedArray<T, simd_V, size_V>> : public std::integral_constant<size_t, size_V * getElementCount<T>(simd_V)> {
};

template <typename T>
inline constexpr size_t elementCount_v = elementCount<T>::value;

namespace alignedArrayImpl
{
class IdentityFormatingFunctor
{
 public:
  template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
  inline std::string operator()(const T& value)
  {
    return fmt::format("{}", value);
  }

  inline std::string operator()(const uint8_t& value)
  {
    return fmt::format("{}", static_cast<uint32_t>(value));
  }
};

class HexFormatingFunctor
{
 public:
  template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
  inline std::string operator()(const T& value)
  {
    return fmt::format("{:#0x}", value);
  }
};

} // namespace alignedArrayImpl

template <typename T, SIMDWidth width_V, size_t size_V, class formater_T = alignedArrayImpl::IdentityFormatingFunctor>
std::ostream& operator<<(std::ostream& stream, const AlignedArray<T, width_V, size_V>& array)
{
  stream << "[";
  for (auto subspan : array) {
    operator<< <const T, getElementCount<T>(width_V), formater_T>(stream, subspan);
    stream << ", ";
  }
  stream << "]";
  return stream;
};

template <typename T, SIMDWidth width_V, size_t size_V>
std::string asHex(const AlignedArray<T, width_V, size_V>& array)
{
  std::ostringstream stream;
  operator<< <T, width_V, size_V, alignedArrayImpl::HexFormatingFunctor>(stream, array);
  return stream.str();
};

} // namespace o2::rans::internal::simd

namespace std
{
template <typename T, size_t extent_V, class formatingFunctor = o2::rans::internal::simd::alignedArrayImpl::IdentityFormatingFunctor>
std::ostream& operator<<(std::ostream& stream, const gsl::span<T, extent_V>& span)
{

  if (span.empty()) {
    stream << "[]";
    return stream;
  } else {
    formatingFunctor formater;

    stream << "[";
    for (size_t i = 0; i < span.size() - 1; ++i) {
      stream << formater(span[i]) << ", ";
    }
    stream << formater(*(--span.end())) << "]";
    return stream;
  }
  return stream;
}
} // namespace std

namespace gsl
{
template <typename T, o2::rans::internal::simd::SIMDWidth width_V, size_t size_V>
auto make_span(const o2::rans::internal::simd::AlignedArray<T, width_V, size_V>& array)
{
  return gsl::span<const T, o2::rans::internal::simd::AlignedArray<T, width_V, size_V>::nElements()>(array.data(), array.nElements());
};

template <typename T, o2::rans::internal::simd::SIMDWidth width_V, size_t size_V>
auto make_span(o2::rans::internal::simd::AlignedArray<T, width_V, size_V>& array)
{
  return gsl::span<T, o2::rans::internal::simd::AlignedArray<T, width_V, size_V>::nElements()>(array.data(), array.nElements());
};

} // namespace gsl

#endif /* RANS_INTERNAL_CONTAINERS_ALIGNEDARRAY_H_ */