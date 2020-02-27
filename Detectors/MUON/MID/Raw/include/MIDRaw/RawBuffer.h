// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDRaw/RawBuffer.h
/// \brief  Handler of the RAW buffer
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   28 November 2019
#ifndef O2_MID_RAWBUFFER_H
#define O2_MID_RAWBUFFER_H

#include <cstdint>
#include <vector>
#include <gsl/gsl>
#include "Headers/RAWDataHeader.h"

namespace o2
{
namespace mid
{
template <typename T>
class RawBuffer
{
 public:
  enum class ResetMode {
    all,            // Reset buffer and indexes
    keepUnconsumed, // Keep the last unconsumed HB
    bufferOnly      // Do not reset indexes
  };

  void setBuffer(gsl::span<const T> bytes, ResetMode resetMode = ResetMode::keepUnconsumed);

  /// Gets the current RDH
  const header::RAWDataHeader* getRDH() { return mRDH; }

  unsigned int next(unsigned int nBits);
  T next();

  bool nextHeader();

  bool hasNext(unsigned int nBytes);

  void skipOverhead();

  bool isHBClosed();

 private:
  gsl::span<const T> mBytes{};                                    /// gsl span with encoded information
  gsl::span<const T> mCurrentBuffer{};                            /// gsl span with the current encoded information
  std::vector<T> mUnconsumed{};                                   /// Unconsumed buffer
  size_t mElementIndex{0};                                        /// Index of the current element in the buffer
  size_t mBitIndex{0};                                            /// Index of the current bit
  size_t mHeaderIndex{0};                                         /// Index of the current header
  size_t mNextHeaderIndex{0};                                     /// Index of the next header
  size_t mEndOfPayloadIndex{0};                                   /// Index of the end of payload
  const unsigned int mElementSizeInBytes{sizeof(T)};              /// Element size in bytes
  const unsigned int mElementSizeInBits{mElementSizeInBytes * 8}; /// Element size in bits
  const header::RAWDataHeader* mRDH{nullptr};                     /// Current header (not owner)

  bool nextPayload();
  void reset();
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_RAWBUFFER_H */
