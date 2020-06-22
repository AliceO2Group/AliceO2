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

#ifndef RANS_HELPER_H
#define RANS_HELPER_H

#include <cstddef>
#include <cmath>
#include <chrono>

namespace o2
{
namespace rans
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

inline size_t calculateMaxBufferSize(size_t num, size_t rangeBits, size_t sizeofStreamT)
{
  // RS: w/o safety margin the o2-test-ctf-io produces an overflow in the Encoder::process
  constexpr size_t SaferyMargin = 16;
  return std::ceil(1.20 * (num * rangeBits * 1.0) / (sizeofStreamT * 8.0)) + SaferyMargin;
}

//rans default values
constexpr size_t ProbabilityBits8Bit = 10;
constexpr size_t ProbabilityBits16Bit = 22;
constexpr size_t ProbabilityBits25Bit = 25;

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

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> mStart;
  std::chrono::time_point<std::chrono::high_resolution_clock> mStop;
};

} // namespace rans
} // namespace o2

#endif /* RANS_HELPER_H */
