// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

inline constexpr size_t numSymbolsWithNBits(size_t bits) noexcept
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

template <typename T, typename IT>
inline constexpr bool isCompatibleIter_v = std::is_convertible_v<typename std::iterator_traits<IT>::value_type, T>;
template <typename IT>
inline constexpr bool isIntegralIter_v = std::is_integral_v<typename std::iterator_traits<IT>::value_type>;

} // namespace internal
} // namespace rans
} // namespace o2

#endif /* RANS_INTERNAL_HELPER_H */
