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

/// @file   utils.h
/// @author Michael Lettrich
/// @brief  common helper classes and functions

#ifndef RANS_INTERNAL_COMMON_UTILS_H_
#define RANS_INTERNAL_COMMON_UTILS_H_

#include <cstddef>
#include <cmath>
#include <chrono>
#include <type_traits>
#include <iterator>
#include <sstream>
#include <vector>
#include <cstring>

#include <fairlogger/Logger.h>

#include "rANS/internal/common/defaults.h"
#include "rANS/internal/common/exceptions.h"

#define rans_likely(x) __builtin_expect((x), 1)
#define rans_unlikely(x) __builtin_expect((x), 0)

namespace o2::rans
{

namespace utils
{

template <typename T>
constexpr size_t toBits() noexcept;

} // namespace utils

namespace internal
{
// taken from https://github.com/romeric/fastapprox/blob/master/fastapprox/src/fastlog.h
[[nodiscard]] inline constexpr float_t fastlog2(float_t x) noexcept
{
  union {
    float_t f;
    uint32_t i;
  } vx = {x};

  union {
    uint32_t i;
    float_t f;
  } mx = {(vx.i & 0x007FFFFF) | 0x3f000000};

  float_t y = vx.i;
  y *= 1.1920928955078125e-7f;

  return y - 124.22551499f - 1.498030302f * mx.f - 1.72587999f / (0.3520887068f + mx.f);
};

template <typename T>
inline size_t itemsPerQWord()
{
  return sizeof(uint64_t) / sizeof(T);
}

inline uint64_t load64(const void* __restrict src)
{
  uint64_t ret;
  std::memcpy(&ret, src, 8);
  return ret;
};

inline void write64(void* __restrict dest, uint64_t src) { std::memcpy(dest, &src, 8); };

template <typename T>
inline constexpr uintptr_t adr2Bits(T* address) noexcept
{
  return (reinterpret_cast<uintptr_t>(address) << 3ull);
};

template <typename T>
inline constexpr T log2UIntNZ(T x) noexcept
{
  static_assert(std::is_integral_v<T>, "Type is not integral");
  static_assert(std::is_unsigned_v<T>, "only defined for unsigned numbers");
  assert(x > 0);

  if constexpr (sizeof(T) <= 4) {
    return static_cast<T>(utils::toBits<uint32_t>() - __builtin_clz(x) - 1);
  } else {
    return static_cast<T>(utils::toBits<uint64_t>() - __builtin_clzl(x) - 1);
  }
}

template <typename T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
inline constexpr bool isPow2(T x) noexcept
{
  return x > 0 && (x & (x - 1)) == 0;
}

[[nodiscard]] inline count_t roundSymbolFrequency(double_t rescaledFrequency)
{
  const count_t roundedDown = static_cast<count_t>(rescaledFrequency);
  const double_t roundedDownD = roundedDown;

  // rescaledFrequency**2 <= ( floor(rescaledFrequency) * ceil(rescaledFrequency))
  if (rescaledFrequency * rescaledFrequency <= (roundedDownD * (roundedDownD + 1.0))) {
    // round down
    return roundedDown;
  } else {
    // round up
    return roundedDown + 1;
  }
};

inline constexpr size_t numSymbolsWithNBits(size_t bits) noexcept
{
  return (static_cast<size_t>(1) << (bits + 1)) - 1;
};

inline constexpr size_t numBitsForNSymbols(size_t nSymbols) noexcept
{
  switch (nSymbols) {
    case 0:
      return 0; // to represent 0 symbols we need 0 bits
      break;
    case 1: // to represent 1 symbol we need 1 bit
      return 1;
      break;
    default: // general case for > 1 symbols
      return std::ceil(std::log2(nSymbols));
      break;
  }
}

inline uint32_t safeadd(uint32_t a, uint32_t b)
{
  uint32_t result;
  if (rans_unlikely(__builtin_uadd_overflow(a, b, &result))) {
    throw OverflowError("arithmetic overflow during addition");
  }
  return result;
}

} // namespace internal

inline constexpr std::uint8_t operator"" _u8(unsigned long long int value) { return static_cast<uint8_t>(value); };
inline constexpr std::int8_t operator"" _i8(unsigned long long int value) { return static_cast<int8_t>(value); };

inline constexpr std::uint16_t operator"" _u16(unsigned long long int value) { return static_cast<uint16_t>(value); };
inline constexpr std::int16_t operator"" _i16(unsigned long long int value) { return static_cast<int16_t>(value); };

namespace utils
{
inline constexpr size_t toBytes(size_t bits) noexcept { return (bits / 8) + (bits % 8 != 0); };

inline constexpr size_t pow2(size_t n) noexcept
{
  return 1ull << n;
}

inline constexpr size_t toBits(size_t bytes) noexcept { return bytes * 8; };

template <typename T>
inline constexpr size_t toBits() noexcept
{
  return toBits(sizeof(T));
};

template <typename T>
inline constexpr T log2UInt(T x) noexcept
{
  static_assert(std::is_integral_v<T>, "Type is not integral");
  static_assert(std::is_unsigned_v<T>, "only defined for unsigned numbers");
  if (x > static_cast<T>(0)) {
    return internal::log2UIntNZ<T>(x);
  } else {
    return static_cast<T>(0);
  }
}

template <typename Freq_IT>
inline Freq_IT advanceIter(Freq_IT iter, std::ptrdiff_t distance)
{
  std::advance(iter, distance);
  return iter;
}

[[nodiscard]] inline uint32_t symbolLengthBits(uint32_t x) noexcept { return log2UInt(x); };

template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
[[nodiscard]] inline constexpr uint32_t getRangeBits(T min, T max) noexcept
{
  assert(max >= min);
  const int64_t diff = max - min;

  if (diff == 0) {
    return 0; // if min==max, we're empty. Compatible with the case that we need 2**0 == 1 Value.
  } else {
    return symbolLengthBits(diff) + 1; // otherwise add 1 to cover full interval [min,max]
  }
};

[[nodiscard]] inline size_t sanitizeRenormingBitRange(size_t renormPrecision)
{
  size_t sanitizedPrecision{};
  if (renormPrecision != 0) {
    sanitizedPrecision = std::min(defaults::MaxRenormPrecisionBits, std::max(defaults::MinRenormPrecisionBits, renormPrecision));
    LOG_IF(debug, (sanitizedPrecision != renormPrecision)) << fmt::format("Renorming precision {} is not in valid interval [{},{}], rounding to {} ",
                                                                          renormPrecision,
                                                                          defaults::MinRenormPrecisionBits,
                                                                          defaults::MaxRenormPrecisionBits,
                                                                          sanitizedPrecision);
  } else {
    // allow 0 as special case to handle empty frequency tables
    sanitizedPrecision = 0;
  }
  return sanitizedPrecision;
};

template <typename T>
[[nodiscard]] inline size_t constexpr nBytesTo(size_t nBytes) noexcept
{
  const size_t nOthers = nBytes / sizeof(T) + (nBytes % sizeof(T) > 0);
  return nOthers;
};

[[nodiscard]] inline constexpr bool isValidRenormingPrecision(size_t renormPrecision)
{
  const bool isInInterval = (renormPrecision >= defaults::MinRenormPrecisionBits) && (renormPrecision <= defaults::MaxRenormPrecisionBits);
  const bool isZeroMessage = renormPrecision == 0;
  return isInInterval || isZeroMessage;
};

template <typename IT>
void checkBounds(IT iteratorPosition, IT upperBound)
{
  const auto diff = std::distance(iteratorPosition, upperBound);
  if (diff < 0) {
    throw OutOfBoundsError(fmt::format("Bounds of buffer violated by {} elements", std::abs(diff)));
  }
}

class RANSTimer
{
 public:
  void start() { mStart = std::chrono::high_resolution_clock::now(); };
  void stop() { mStop = std::chrono::high_resolution_clock::now(); };
  inline double getDurationMS() noexcept
  {
    std::chrono::duration<double, std::milli> duration = mStop - mStart;
    return duration.count();
  }

  inline double getDurationS() noexcept
  {
    std::chrono::duration<double, std::ratio<1>> duration = mStop - mStart;
    return duration.count();
  }

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> mStart;
  std::chrono::time_point<std::chrono::high_resolution_clock> mStop;
};

template <class T>
class JSONArrayLogger
{
 public:
  explicit JSONArrayLogger(bool reverse = false) : mReverse{reverse} {};
  inline JSONArrayLogger& operator<<(const T& elem)
  {
    mElems.emplace_back(elem);
    return *this;
  };

  friend std::ostream& operator<<(std::ostream& os, const JSONArrayLogger& logger)
  {
    auto printSymbols = [&](auto begin, auto end) {
      --end;
      for (auto it = begin; it != end; ++it) {
        os << +static_cast<T>(*it) << " ,";
      }
      os << +static_cast<T>(*end);
    };

    os << "[";
    if (!logger.mElems.empty()) {
      if (logger.mReverse) {
        printSymbols(logger.mElems.rbegin(), logger.mElems.rend());
      } else {
        printSymbols(logger.mElems.begin(), logger.mElems.end());
      }
    }
    os << "]";
    return os;
  }

 private:
  std::vector<T> mElems{};
  bool mReverse{false};
};

template <typename T, typename IT>
inline constexpr bool isCompatibleIter_v = std::is_convertible_v<typename std::iterator_traits<IT>::value_type, T>;

template <typename IT>
inline constexpr bool isIntegralIter_v = std::is_integral_v<typename std::iterator_traits<IT>::value_type>;

} // namespace utils

} // namespace o2::rans

#endif /* RANS_INTERNAL_COMMON_UTILS_H_ */
