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

namespace o2
{
namespace mid
{
class ELinkDecoder
{
 public:
  bool add(const uint8_t byte, uint8_t expectedStart);
  /// Checks if we have all of the information needed for the decoding
  bool isComplete() const { return mBytes.size() == mTotalSize; };
  /// Gets the status word
  uint8_t getStatusWord() const { return mBytes[0]; }
  /// Gets the event word
  uint8_t getEventWord() const { return mBytes[1]; }
  /// Gets the counter
  uint16_t getCounter() const { return joinBytes(2); }
  // uint16_t getCounter() const { return (mBytes[2] << 8) | mBytes[3]; }
  /// Gets the card ID
  uint8_t getId() const { return (mBytes[4] >> 4) & 0xF; }
  /// Gets the inputs
  uint8_t getInputs() const { return (mBytes[4] & 0xF); }
  uint16_t getPattern(int cathode, int chamber) const;
  /// Gets the number of bytes read
  size_t getNBytes() const { return mBytes.size(); }

  void reset();

 private:
  inline uint16_t joinBytes(int idx) const { return (mBytes[idx] << 8 | mBytes[idx + 1]); };

  static constexpr size_t sMinimumSize{5};  /// Minimum size of the buffer
  static constexpr size_t sMaximumSize{21}; /// Maximum size of the buffer
  std::vector<uint8_t> mBytes{};            /// Vector with encoded information
  size_t mTotalSize{sMinimumSize};          /// Expected size of the read-out buffer
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_ELINKDECODER_H */
