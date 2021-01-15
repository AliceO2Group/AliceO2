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
inline constexpr bool needs64Bit()
{
  return sizeof(T) > 4;
}

inline constexpr size_t bitsToRange(size_t bits)
{
  return 1 << bits;
}

class RANSTimer
{
 public:
  void start() { mStart = std::chrono::high_resolution_clock::now(); };
  void stop() { mStop = std::chrono::high_resolution_clock::now(); };
  double getDurationMS()
  {
    std::chrono::duration<double, std::milli> duration = mStop - mStart;
    return duration.count();
  }

  double getDurationS()
  {
    std::chrono::duration<double, std::ratio<1>> duration = mStop - mStart;
    return duration.count();
  }

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> mStart;
  std::chrono::time_point<std::chrono::high_resolution_clock> mStop;
};

template <typename T, typename IT>
inline constexpr bool isCompatibleIter_v = std::is_same_v<typename std::iterator_traits<IT>::value_type, T>;

} // namespace internal
} // namespace rans
} // namespace o2

#endif /* RANS_INTERNAL_HELPER_H */
