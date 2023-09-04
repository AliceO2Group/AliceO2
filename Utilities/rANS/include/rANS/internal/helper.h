// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   helper.h
/// @author Michael Lettrich
/// @since  2019-05-21
/// @brief  various helper functions

#ifndef RANS_INTERNAL_HELPER_H
#define RANS_INTERNAL_HELPER_H

#include <cstddef>
#include <cmath>
#include <chrono>
#include <type_traits>
#include <iterator>
#include <sstream>
#include <vector>

#define rans_likely(x) __builtin_expect((x), 1)
#define rans_unlikely(x) __builtin_expect((x), 0)

namespace o2
{
namespace rans
{
namespace internal
{

template <typename T>
inline constexpr bool needs64Bit() noexcept
{
  return sizeof(T) > 4;
}

inline constexpr size_t pow2(size_t n) noexcept
{
  return 1 << n;
}

inline constexpr uint32_t log2UInt(uint32_t x) noexcept
{
  return x > 0 ? sizeof(int) * 8 - __builtin_clz(x) - 1 : 0;
}

template <typename T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
inline constexpr bool isPow2(T x) noexcept
{
  return x > 0 && (x & (x - 1)) == 0;
}

inline constexpr size_t
  numSymbolsWithNBits(size_t bits) noexcept
{
  return (1 << (bits + 1)) - 1;
}

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

template <typename Freq_IT>
inline Freq_IT advanceIter(Freq_IT iter, std::ptrdiff_t distance)
{
  std::advance(iter, distance);
  return iter;
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

inline uint32_t safeadd(uint32_t a, uint32_t b)
{
  uint32_t result;
  if (rans_unlikely(__builtin_uadd_overflow(a, b, &result))) {
    throw std::overflow_error("arithmetic overflow during addition");
  }
  return result;
}

template <typename T, typename IT>
inline constexpr bool isCompatibleIter_v = std::is_convertible_v<typename std::iterator_traits<IT>::value_type, T>;
template <typename IT>
inline constexpr bool isIntegralIter_v = std::is_integral_v<typename std::iterator_traits<IT>::value_type>;

} // namespace internal
} // namespace rans
} // namespace o2

#endif /* RANS_INTERNAL_HELPER_H */
