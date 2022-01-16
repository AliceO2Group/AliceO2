// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   tyspes.h
/// @author Michael Lettrich
/// @since  2021-06-12
/// @brief

#ifndef RANS_INTERNAL_SIMD_TYPES_H
#define RANS_INTERNAL_SIMD_TYPES_H

#include <cstring>
#include <cassert>
#include <cstdint>
#include <type_traits>

#include <sstream>
#include <fmt/format.h>

#include "rANS/internal/backend/simd/utils.h"
#include "rANS/internal/helper.h"

namespace o2
{
namespace rans
{
namespace internal
{
namespace simd
{

enum class SIMDWidth : uint32_t { SSE = 128u,
                                  AVX = 256u };

inline constexpr size_t getLaneWidthBits(SIMDWidth width) noexcept { return static_cast<size_t>(width); };

inline constexpr size_t getLaneWidthBytes(SIMDWidth width) noexcept { return toBytes(static_cast<size_t>(width)); };

inline constexpr size_t getAlignment(SIMDWidth width) noexcept { return getLaneWidthBytes(width); };

template <class T, size_t N>
inline constexpr T* assume_aligned(T* ptr) noexcept
{
  return reinterpret_cast<T*>(__builtin_assume_aligned(ptr, N, 0));
};

template <class T, SIMDWidth width_V>
inline constexpr T* assume_aligned(T* ptr) noexcept
{
  constexpr size_t alignment = getAlignment(width_V);
  return assume_aligned<T, alignment>(ptr);
};

template <typename T, SIMDWidth width_V>
inline constexpr bool isAligned(T* ptr)
{
  // only aligned iff ptr is divisible by alignment
  constexpr size_t alignment = getAlignment(width_V);
  return !(reinterpret_cast<uintptr_t>(ptr) % alignment);
};

template <typename T>
inline constexpr size_t getElementCount(SIMDWidth width) noexcept
{
  return getLaneWidthBytes(width) / sizeof(T);
};

template <typename T>
inline constexpr SIMDWidth getSimdWidth(size_t nHardwareStreams) noexcept
{
  return static_cast<SIMDWidth>(nHardwareStreams * toBits(sizeof(T)));
};

template <typename T, SIMDWidth width_V, size_t size_V = getElementCount<T>(width_V)>
class alignas(getAlignment(width_V)) AlignedArray
{
 public:
  using value_type = T;
  using pointer = T*;
  using reference = T&;
  using iterator = T*;
  using const_iterator = const T*;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  static inline constexpr size_t size() noexcept { return size_V; };

  inline constexpr AlignedArray() noexcept {}; //NOLINT

  template <typename elem_T, std::enable_if_t<std::is_convertible_v<elem_T, value_type>, bool> = true>
  inline constexpr AlignedArray(elem_T elem) noexcept
  {
    //#pragma omp simd
    for (size_t i = 0; i < size(); ++i) {
      mData[i] = static_cast<value_type>(elem);
    }
  };

  template <typename... Args, std::enable_if_t<(sizeof...(Args) == AlignedArray<T, width_V, size_V>::size()) && std::is_convertible_v<std::common_type_t<Args...>, value_type>, bool> = true>
  inline constexpr AlignedArray(Args... args) noexcept : mData{static_cast<value_type>(args)...} {};

  inline constexpr const T* data() const noexcept { return mData; };
  inline constexpr T* data() noexcept { return const_cast<T*>(static_cast<const AlignedArray&>(*this).data()); };
  inline constexpr const_iterator begin() const noexcept { return data(); };
  inline constexpr const_iterator end() const noexcept { return data() + size(); };
  inline constexpr iterator begin() noexcept { return const_cast<iterator>(static_cast<const AlignedArray&>(*this).begin()); };
  inline constexpr iterator end() noexcept { return const_cast<iterator>(static_cast<const AlignedArray&>(*this).end()); };
  inline constexpr const_reverse_iterator rbegin() const noexcept { return std::reverse_iterator(this->end()); };
  inline constexpr const_reverse_iterator rend() const noexcept { return std::reverse_iterator(this->begin()); };
  inline constexpr reverse_iterator rbegin() noexcept { return std::reverse_iterator(this->end()); };
  inline constexpr reverse_iterator rend() noexcept { return std::reverse_iterator(this->begin()); };
  inline constexpr const T& operator[](size_t i) const
  {
    assert(i < size());
    return mData[i];
  };
  inline constexpr T& operator[](size_t i) { return const_cast<T&>(static_cast<const AlignedArray&>(*this)[i]); };

 private:
  T mData[size_V]{};
};

namespace impl
{
template <typename From, typename To>
struct isArrayConvertible : public std::bool_constant<std::is_convertible_v<From (*)[], To (*)[]>> {
};

template <typename From, typename To>
inline constexpr bool isArrayConvertible_v = isArrayConvertible<From, To>::value;

} // namespace impl

template <class element_T, size_t extent_V>
class ArrayView
{
 public:
  using element_type = element_T;
  using value_type = std::remove_const_t<element_type>;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer = element_type*;
  using const_pointer = const element_type*;
  using reference = element_type&;
  using const_reference = const element_type&;
  using iterator = element_type*;
  using reverse_iterator = std::reverse_iterator<iterator>;

  inline constexpr ArrayView(pointer ptr, size_t size) noexcept : mBegin{ptr} { assert(size == extent_V); };

  inline constexpr ArrayView(element_type (&array)[extent_V]) noexcept : mBegin{array} {};

  template <typename T, std::enable_if_t<impl::isArrayConvertible_v<T, element_type>, bool> = true>
  inline constexpr ArrayView(std::array<T, extent_V>& array) noexcept : mBegin{static_cast<pointer>(array.data())} {};

  template <class T, std::enable_if_t<impl::isArrayConvertible_v<const T, element_type>, bool> = true>
  inline constexpr ArrayView(const std::array<T, extent_V>& array) noexcept : mBegin{static_cast<pointer>(array.data())} {};

  template <class T, SIMDWidth width_V, std::enable_if_t<impl::isArrayConvertible_v<T, element_type>, bool> = true>
  inline constexpr ArrayView(AlignedArray<T, width_V, extent_V>& array) noexcept : mBegin{static_cast<pointer>(array.data())} {};

  template <class T, SIMDWidth width_V, std::enable_if_t<impl::isArrayConvertible_v<const T, element_type>, bool> = true>
  inline constexpr ArrayView(const AlignedArray<T, width_V, extent_V>& array) noexcept : mBegin{static_cast<pointer>(array.data())} {};

  inline constexpr iterator begin() const noexcept
  {
    return mBegin;
  };
  inline constexpr iterator end() const noexcept { return mBegin + this->size(); };
  inline constexpr reverse_iterator rbegin() const noexcept { return std::reverse_iterator(this->end()); };
  inline constexpr reverse_iterator rend() const noexcept { return std::reverse_iterator(this->begin()); };

  inline constexpr pointer data() const noexcept { return mBegin; };
  inline constexpr reference operator[](size_t index) const
  {
    assert(index < size());
    return *(mBegin + index);
  };

  inline constexpr size_type size() const noexcept { return extent_V; };
  inline constexpr bool empty() const noexcept { return this->size() == 0ull; };

  template <size_t offset_V, size_t count_V>
  inline constexpr ArrayView<element_type, count_V> subView() const noexcept
  {
    static_assert(count_V <= extent_V);
    static_assert(count_V <= (extent_V - offset_V));
    return ArrayView<element_type, count_V>{mBegin + offset_V, count_V};
  };

 private:
  pointer mBegin{};
};

// Deduction Guides
template <class T, std::size_t size_V>
ArrayView(T (&)[size_V]) -> ArrayView<T, size_V>;

template <class T, std::size_t size_V>
ArrayView(std::array<T, size_V>&) -> ArrayView<T, size_V>;

template <class T, std::size_t size_V>
ArrayView(const std::array<T, size_V>&) -> ArrayView<const T, size_V>;

template <class T, SIMDWidth width_V, size_t extent_V>
ArrayView(AlignedArray<T, width_V, extent_V>&) -> ArrayView<T, extent_V>;

template <class T, SIMDWidth width_V, size_t extent_V>
ArrayView(const AlignedArray<T, width_V, extent_V>&) -> ArrayView<const T, extent_V>;

namespace impl
{

template <typename T>
inline constexpr size_t simdViewSize(SIMDWidth width, size_t extent) noexcept
{
  return getElementCount<T>(width) * extent;
}

} // namespace impl

template <class element_T, SIMDWidth width_V, size_t extent_V, bool aligned_V = true>
class SIMDView
{

 public:
  class Iterator;

 public:
  using element_type = element_T;
  using value_type = std::remove_const_t<element_type>;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer = element_type*;
  using const_pointer = const element_type*;
  using reference = element_type&;
  using const_reference = const element_type&;
  using iterator = SIMDView::Iterator;
  using reverse_iterator = std::reverse_iterator<iterator>;

 public:
  inline constexpr SIMDView(pointer ptr, size_t size) : mBegin{ptr}
  {
    assert(size == extent_V);
  };

  inline constexpr SIMDView(element_type (&array)[impl::simdViewSize<element_type>(width_V, extent_V)]) : mBegin{array} { checkAlignment(mBegin); };

  template <typename T, std::enable_if_t<impl::isArrayConvertible_v<T, element_type>, bool> = true>
  inline constexpr SIMDView(std::array<T, impl::simdViewSize<element_type>(width_V, extent_V)>& array) : mBegin{static_cast<pointer>(array.data())}
  {
    checkAlignment(mBegin);
  };

  template <class T, std::enable_if_t<impl::isArrayConvertible_v<const T, element_type>, bool> = true>
  inline constexpr SIMDView(const std::array<T, impl::simdViewSize<element_type>(width_V, extent_V)>& array) : mBegin{static_cast<pointer>(array.data())}
  {
    checkAlignment(mBegin);
  };

  template <class T, size_t size_V, std::enable_if_t<impl::isArrayConvertible_v<T, element_type>, bool> = true>
  inline constexpr SIMDView(AlignedArray<T, width_V, size_V>& array) : mBegin{static_cast<pointer>(array.data())}
  {
    static_assert(size_V == impl::simdViewSize<element_type>(width_V, extent_V));
  };

  template <class T, size_t size_V, std::enable_if_t<impl::isArrayConvertible_v<const T, element_type>, bool> = true>
  inline constexpr SIMDView(const AlignedArray<T, width_V, size_V>& array) : mBegin{static_cast<pointer>(array.data())}
  {
    static_assert(size_V == impl::simdViewSize<element_type>(width_V, extent_V));
  };

  inline constexpr iterator begin() const noexcept { return iterator{mBegin}; };
  inline constexpr iterator end() const noexcept { return iterator{mBegin + impl::simdViewSize<element_type>(width_V, extent_V)}; };
  inline constexpr reverse_iterator rbegin() const noexcept { return std::reverse_iterator(this->end()); };
  inline constexpr reverse_iterator rend() const noexcept { return std::reverse_iterator(this->begin()); };

  inline constexpr pointer data() const noexcept
  {
    if constexpr (aligned_V) {
      return assume_aligned<element_type, width_V>(mBegin);
    } else {
      return mBegin;
    }
  };
  inline constexpr reference operator[](size_t index) const
  {
    assert(index < extent_V);
    return mBegin[index * getElementCount<element_type>(width_V)];
  };

  inline constexpr size_t size() const noexcept { return extent_V; };
  inline constexpr bool empty() const noexcept { return this->size() == 0ull; };

  template <size_t offset_V, size_t count_V>
  inline constexpr SIMDView<element_type, width_V, count_V, aligned_V> subView() const
  {
    static_assert(count_V <= extent_V);
    static_assert(count_V <= (extent_V - offset_V));
    return SIMDView<element_type, width_V, count_V, aligned_V>{mBegin + (offset_V * getElementCount<element_type>(width_V)), count_V};
  };

  inline constexpr operator ArrayView<element_type, impl::simdViewSize<element_type>(width_V, extent_V)>() const noexcept
  {
    return {mBegin, impl::simdViewSize<element_type>(width_V, extent_V)};
  };

  inline constexpr operator SIMDView<element_T, width_V, extent_V, false>() const
  {
    if constexpr (aligned_V) {
      return {mBegin, extent_V};
    } else {
      return *this;
    }
  };

  inline constexpr operator SIMDView<element_T, width_V, extent_V, true>() const
  {
    if constexpr (aligned_V) {
      return *this;
    } else {
      return {mBegin, extent_V};
    }
  };

 private:
  template <typename T>
  inline constexpr void checkAlignment(T* ptr)
  {
    // material implication aligned_V -> isAligned(ptr)
    const bool alignment = !aligned_V || isAligned<T, width_V>(ptr);
    if (__builtin_expect(!alignment, 0)) {
      throw std::runtime_error("alignment missmatch");
    }
  };

  pointer mBegin{};

 public:
  class Iterator
  {
   public:
    using difference_type = std::ptrdiff_t;
    using value_type = SIMDView::element_type;
    using pointer = value_type*;
    using reference = value_type&;
    using iterator_category = std::random_access_iterator_tag;

    inline constexpr Iterator() noexcept = default;
    inline constexpr explicit Iterator(SIMDView::pointer ptr) noexcept : mIter{ptr} {};
    inline constexpr Iterator(const Iterator& iter) noexcept = default;
    inline constexpr Iterator(Iterator&& iter) noexcept = default;
    inline constexpr Iterator& operator=(const Iterator& other) noexcept = default;
    inline constexpr Iterator& operator=(Iterator&& other) noexcept = default;
    inline ~Iterator() noexcept = default;

    // pointer arithmetics
    inline constexpr Iterator& operator++() noexcept
    {
      mIter += advanceBySIMDStep(1);
      return *this;
    };

    inline constexpr Iterator operator++(int) noexcept
    {
      auto res = *this;
      ++(*this);
      return res;
    };

    inline constexpr Iterator& operator--() noexcept
    {
      mIter -= advanceBySIMDStep(1);
      return *this;
    };

    inline constexpr Iterator operator--(int) noexcept
    {
      auto res = *this;
      --(*this);
      return res;
    };

    inline constexpr Iterator& operator+=(difference_type i) noexcept
    {
      mIter += advanceBySIMDStep(i);
      return *this;
    };

    inline constexpr Iterator operator+(difference_type i) const noexcept
    {
      auto tmp = *const_cast<Iterator*>(this);
      return tmp += advanceBySIMDStep(i);
    }

    inline constexpr Iterator& operator-=(difference_type i) noexcept
    {
      mIter -= advanceBySIMDStep(i);
      return *this;
    };

    inline constexpr Iterator operator-(difference_type i) const noexcept
    {
      auto tmp = *const_cast<Iterator*>(this);
      return tmp -= advanceBySIMDStep(i);
    };

    inline constexpr difference_type operator-(const Iterator& other) const noexcept
    {
      return this->mIter - other.mIter;
    };

    // comparison
    inline constexpr bool operator==(const Iterator& other) const noexcept { return this->mIter == other.mIter; };
    inline constexpr bool operator!=(const Iterator& other) const noexcept { return this->mIter != other.mIter; };
    inline constexpr bool operator<(const Iterator& other) const noexcept { return this->mIter < other->mIter; };
    inline constexpr bool operator>(const Iterator& other) const noexcept { return this->mIter > other->mIter; };
    inline constexpr bool operator>=(const Iterator& other) const noexcept { return this->mIter >= other->mIter; };
    inline constexpr bool operator<=(const Iterator& other) const noexcept { return this->mIter <= other->mIter; };

    // dereference
    inline constexpr const value_type& operator*() const { return *mIter; };
    inline constexpr value_type& operator*() { return *mIter; };

    inline constexpr const value_type& operator[](difference_type i) const noexcept { return *(*this + i); };

   private:
    pointer mIter{};

    inline constexpr difference_type advanceBySIMDStep(difference_type nSteps) noexcept
    {
      return getElementCount<element_type>(width_V) * nSteps;
    };
  };
};

// seems not to work with GCC 10.x, so provide factory functions instead
// //Type deduction guides
// template <class T, SIMDWidth width_V, size_t extent_V>
// SIMDView(AlignedArray<T, width_V, extent_V>&) -> SIMDView<T, width_V, (extent_V / getElementCount<T>(width_V)), true>;

// template <class T, SIMDWidth width_V, size_t extent_V>
// SIMDView(const AlignedArray<T, width_V, extent_V>&) -> SIMDView<const T, width_V, (extent_V / getElementCount<T>(width_V)), true>;

template <typename T, SIMDWidth width_V, size_t extent_V>
inline constexpr SIMDView<T, width_V, (extent_V / getElementCount<T>(width_V)), true> toSIMDView(AlignedArray<T, width_V, extent_V>& array) noexcept
{
  return {array};
}

template <typename T, SIMDWidth width_V, size_t extent_V>
inline constexpr SIMDView<const T, width_V, (extent_V / getElementCount<T>(width_V)), true> toConstSIMDView(const AlignedArray<T, width_V, extent_V>& array) noexcept
{
  return {array};
}

template <typename T>
struct simdWidth;

template <typename T, SIMDWidth simd_V>
struct simdWidth<AlignedArray<T, simd_V>> : public std::integral_constant<SIMDWidth, simd_V> {
};

template <typename T>
inline constexpr SIMDWidth simdWidth_v = simdWidth<T>::value;

template <typename T>
struct elementCount;

template <typename T, SIMDWidth simd_V>
struct elementCount<AlignedArray<T, simd_V>> : public std::integral_constant<size_t, getElementCount<T>(simd_V)> {
};

template <typename T>
inline constexpr size_t elementCount_v = elementCount<T>::value;

template <SIMDWidth width_V, size_t size_V = 1>
using pd_t = AlignedArray<double_t, width_V, getElementCount<double_t>(width_V) * size_V>;
template <SIMDWidth width_V, size_t size_V = 1>
using epi64_t = AlignedArray<uint64_t, width_V, getElementCount<uint64_t>(width_V) * size_V>;
template <SIMDWidth width_V, size_t size_V = 1>
using epi32_t = AlignedArray<uint32_t, width_V, getElementCount<uint32_t>(width_V) * size_V>;
template <SIMDWidth width_V, size_t size_V = 1>
using epi16_t = AlignedArray<uint16_t, width_V, getElementCount<uint16_t>(width_V) * size_V>;
template <SIMDWidth width_V, size_t size_V = 1>
using epi8_t = AlignedArray<uint8_t, width_V, getElementCount<uint8_t>(width_V) * size_V>;

template <SIMDWidth width_V, size_t size_V = 1>
using pdV_t = SIMDView<double_t, width_V, size_V>;
template <SIMDWidth width_V, size_t size_V = 1>
using epi64V_t = SIMDView<uint64_t, width_V, size_V>;
template <SIMDWidth width_V, size_t size_V = 1>
using epi32V_t = SIMDView<uint32_t, width_V, size_V>;
template <SIMDWidth width_V, size_t size_V = 1>
using epi16V_t = SIMDView<uint16_t, width_V, size_V>;
template <SIMDWidth width_V, size_t size_V = 1>
using epi8V_t = SIMDView<uint8_t, width_V, size_V>;

template <SIMDWidth width_V, size_t size_V = 1>
using pdcV_t = SIMDView<const double_t, width_V, size_V>;
template <SIMDWidth width_V, size_t size_V = 1>
using epi64cV_t = SIMDView<const uint64_t, width_V, size_V>;
template <SIMDWidth width_V, size_t size_V = 1>
using epi32cV_t = SIMDView<const uint32_t, width_V, size_V>;
template <SIMDWidth width_V, size_t size_V = 1>
using epi16cV_t = SIMDView<const uint16_t, width_V, size_V>;
template <SIMDWidth width_V, size_t size_V = 1>
using epi8cV_t = SIMDView<const uint8_t, width_V, size_V>;

template <SIMDWidth width_V, size_t size_V>
using pdVec_t = AlignedArray<double_t, width_V, size_V>;
template <SIMDWidth width_V, size_t size_V>
using u64Vec_t = AlignedArray<uint64_t, width_V, size_V>;
template <SIMDWidth width_V, size_t size_V>
using u32Vec_t = AlignedArray<uint32_t, width_V, size_V>;
template <SIMDWidth width_V, size_t size_V>
using u16Vec_t = AlignedArray<uint16_t, width_V, size_V>;
template <SIMDWidth width_V, size_t size_V>
using u8Vec_t = AlignedArray<uint8_t, width_V, size_V>;

inline constexpr std::uint8_t operator"" _u8(unsigned long long int value) { return static_cast<uint8_t>(value); };
inline constexpr std::int8_t operator"" _i8(unsigned long long int value) { return static_cast<int8_t>(value); };

inline constexpr std::uint16_t operator"" _u16(unsigned long long int value) { return static_cast<uint16_t>(value); };
inline constexpr std::int16_t operator"" _i16(unsigned long long int value) { return static_cast<int16_t>(value); };

template <SIMDWidth>
struct toSIMDintType;

template <>
struct toSIMDintType<SIMDWidth::SSE> {
  using value_type = __m128i;
};

template <>
struct toSIMDintType<SIMDWidth::AVX> {
  using value_type = __m256i;
};

template <SIMDWidth width_V>
using toSIMDintType_t = typename toSIMDintType<width_V>::value_type;

template <SIMDWidth>
struct toSIMDdoubleType;

template <>
struct toSIMDdoubleType<SIMDWidth::SSE> {
  using value_type = __m128d;
};

template <>
struct toSIMDdoubleType<SIMDWidth::AVX> {
  using value_type = __m256d;
};

template <SIMDWidth width_V>
using toSIMDdoubleType_t = typename toSIMDdoubleType<width_V>::value_type;

// alignment atributes cause gcc warnings, but we don't need them, so disable for this specific case.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
template <typename T>
struct toSIMDWidth;

template <>
struct toSIMDWidth<__m128> : public std::integral_constant<SIMDWidth, SIMDWidth::SSE> {
};
template <>
struct toSIMDWidth<__m128i> : public std::integral_constant<SIMDWidth, SIMDWidth::SSE> {
};
template <>
struct toSIMDWidth<__m128d> : public std::integral_constant<SIMDWidth, SIMDWidth::SSE> {
};

template <>
struct toSIMDWidth<__m256> : public std::integral_constant<SIMDWidth, SIMDWidth::AVX> {
};
template <>
struct toSIMDWidth<__m256i> : public std::integral_constant<SIMDWidth, SIMDWidth::AVX> {
};
template <>
struct toSIMDWidth<__m256d> : public std::integral_constant<SIMDWidth, SIMDWidth::AVX> {
};

template <typename T>
inline constexpr SIMDWidth toSIMDWidth_v = toSIMDWidth<T>::value;

#pragma GCC diagnostic pop

template <typename T, SIMDWidth width_V, size_t size_V>
std::ostream& operator<<(std::ostream& stream, const AlignedArray<T, width_V, size_V>& vec)
{
  stream << "[";
  for (const auto& elem : vec) {
    stream << elem << ", ";
  }
  stream << "]";
  return stream;
};

template <SIMDWidth width_V, size_t size_V>
std::ostream& operator<<(std::ostream& stream, const AlignedArray<uint8_t, width_V, size_V>& vec)
{
  stream << "[";
  for (const auto& elem : vec) {
    stream << static_cast<uint32_t>(elem) << ", ";
  }
  stream << "]";
  return stream;
};

template <typename T, SIMDWidth width_V, size_t size_V>
std::string asHex(const AlignedArray<T, width_V, size_V>& vec)
{
  std::stringstream ss;
  ss << "[";
  for (const auto& elem : vec) {
    ss << fmt::format("{:#0x}, ", elem);
  }
  ss << "]";
  return ss.str();
};

} // namespace simd
} // namespace internal
} // namespace rans
} // namespace o2

#endif /* RANS_INTERNAL_SIMD_TYPES_H */