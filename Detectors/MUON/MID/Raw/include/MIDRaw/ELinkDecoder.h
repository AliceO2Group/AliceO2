// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDRaw/ELinkDecoder.h
/// \brief  MID e-link decoder
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   26 November 2019
#ifndef O2_MID_ELINKDECODER_H
#define O2_MID_ELINKDECODER_H

#include <cstdint>
#include <vector>
#include <gsl/gsl>

#include "MIDRaw/LocalBoardRO.h"

namespace o2
{
namespace mid
{
class ELinkDecoder
{
 public:
  /// Adds a byte
  inline void add(const uint8_t byte) { mBytes.emplace_back(byte); }
  void addAndComputeSize(const uint8_t byte);
  template <class ITERATOR>
  bool addCore(ITERATOR& it, const ITERATOR& end)
  {
    /// Adds the first 5 bytes
    auto remaining = sMinimumSize - mBytes.size();
    return add(it, end, remaining);
  }
  template <class ITERATOR>
  bool add(ITERATOR& it, const ITERATOR& end)
  {
    /// Adds the board bytes
    auto remaining = mTotalSize - mBytes.size();
    if (add(it, end, remaining)) {
      if (mTotalSize == sMinimumSize) {
        computeSize();
        remaining = mTotalSize - mBytes.size();
        if (remaining) {
          return add(it, end, remaining);
        }
      }
      return true;
    }
    return false;
  }

  /// Adds the first 5 bytes
  inline bool addCore(size_t& idx, gsl::span<const uint8_t> payload, size_t step) { return add(idx, payload, sMinimumSize - mBytes.size(), step); }

  bool add(size_t& idx, gsl::span<const uint8_t> payload, size_t step);

  /// Checks if this is a zero
  inline bool isZero(uint8_t byte) const { return (mBytes.empty() && (byte & raw::sSTARTBIT) == 0); }

  /// Checks if we have all of the information needed for the decoding
  inline bool isComplete() const { return mBytes.size() == mTotalSize; };
  /// Gets the status word
  inline uint8_t getStatusWord() const { return mBytes[0]; }
  /// Gets the trigger word
  inline uint8_t getTriggerWord() const { return mBytes[1]; }
  /// Gets the counter
  inline uint16_t getCounter() const { return joinBytes(2); }
  /// Gets the card ID
  inline uint8_t getId() const { return (mBytes[4] >> 4) & 0xF; }
  /// Gets the inputs
  inline uint8_t getInputs() const { return (mBytes[4] & 0xF); }
  uint16_t getPattern(int cathode, int chamber) const;
  /// Gets the number of bytes read
  inline size_t getNBytes() const { return mBytes.size(); }

  void reset();

 private:
  inline uint16_t joinBytes(int idx) const { return (mBytes[idx] << 8 | mBytes[idx + 1]); };
  template <class ITERATOR>
  bool add(ITERATOR& it, const ITERATOR& end, size_t nBytes)
  {
    /// Fills inner bytes vector
    auto nToEnd = std::distance(it, end);
    auto nAdded = nBytes < nToEnd ? nBytes : nToEnd;
    mBytes.insert(mBytes.end(), it, it + nAdded);
    it += nAdded;
    return (nAdded == nBytes);
  }

  bool add(size_t& idx, gsl::span<const uint8_t> payload, size_t nBytes, size_t step);

  void computeSize();

  static constexpr size_t sMinimumSize{5};  /// Minimum size of the buffer
  static constexpr size_t sMaximumSize{21}; /// Maximum size of the buffer
  std::vector<uint8_t> mBytes{};            /// Vector with encoded information
  size_t mTotalSize{sMinimumSize};          /// Expected size of the read-out buffer
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_ELINKDECODER_H */
