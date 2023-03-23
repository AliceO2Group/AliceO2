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

#ifndef O2_DATAFORMATS_MCH_CHANNEL_CODE_H_
#define O2_DATAFORMATS_MCH_CHANNEL_CODE_H_

#include <RtypesCore.h>
#include <cstdint>
#include <string>
#include "Rtypes.h"

namespace o2::mch
{
/** 64-bits identifier of a MCH channel.
 *
 * The ChannelCode class encodes in a 64 bits integer the following
 * information about a MCH channel :
 *
 *  - detection element identifier (and index)
 *  - pad index within the detection element
 *  - dual sampa identifier (and index)
 *  - solar identifier (and index)
 *  - elink identifier (or index, those are the same)
 *  - channel number
 *
 * This class serves the same purposes as @ref DsChannelId, but using
 * two different "coordinate systems" to reference the elements
 * within the spectrometer, while DsChannelId uses only one.
 * @ref DsChannelId is readout/online oriented,
 * while ChannelCode is both reconstruction/offline _and_
 * readout/online oriented.
 * But at a cost of twice the size.
 *
 * Note that while this class is internal storing _indices-,
 * it also offer getters for _identifiers_ (Ids).
 *
 */
class ChannelCode
{
 public:
  ChannelCode() = default;
  /** Ctor using "detector oriented" numbering.
   *
   * @param deId detection element identifier (e.g. 100, 502, 1025)
   * @param dePadIndex pad index within the detection element [0..(npads in DE)-1]
   *
   * @throw runtime_error if (deId,dePadIndex) is not a valid combination
   */
  ChannelCode(uint16_t deId, uint16_t dePadIndex);
  /** Ctor using "electronics oriented" numbering.
   *
   * @param solarId solar identifier
   * @param elinkIndex elink index 0..39
   * @param channel channel number 0..63
   *
   * @throw runtime_error if (solarId,elinkIndex,channel) is not a valid combination
   *
   * Note that elinkIndex is also called elinkId equivalently
   * in some other parts of the code, as for elink the id is the index.
   */
  ChannelCode(uint16_t solarId, uint8_t elinkIndex, uint8_t channel);

  /** return the detection element _identifier_ */
  uint16_t getDeId() const;
  /** return the pad _index_ within the detection element.
   * It's in the range 0..(npads in DE)-1) */
  uint16_t getDePadIndex() const;
  /** return the dual sampa _identifier_ */
  uint16_t getDsId() const;
  /** return the dual sampa _index_ (0..16819) */
  uint16_t getDsIndex() const;
  /** return the solar _identifier_ */
  uint16_t getSolarId() const;
  /** return the solar _index_ (0..623) */
  uint16_t getSolarIndex() const;
  /** return the dual sampa channel (0..63) */
  uint8_t getChannel() const;
  /** return the detection element _index_ (0..155) */
  uint8_t getDeIndex() const;
  /** return the elink identifier = index (0..39) */
  uint8_t getElinkId() const;
  /** return the elink index = identifier (0..39) */
  uint8_t getElinkIndex() const { return getElinkId(); }

  /* whether the code is valid.
   *
   * Note that the only way to build an invalid ChannelCode is by using
   * the default constructor, that we cannot suppress
   * (needed e.g. by Root or by some vector functions)
   */
  bool isValid() const { return mValue != sInvalidValue; }

  /* get the actual code */
  uint64_t value() const { return mValue; }

 private:
  void set(uint8_t deIndex,
           uint16_t dePadIndex,
           uint16_t dsIndex,
           uint16_t solarIndex,
           uint8_t elinkIndex,
           uint8_t channel);

 private:
  /* marker for an invalid value. */
  static const uint64_t sInvalidValue{0xFFFFFFFF};

  /** mValue content :
   *
   * - deIndex (0..155) -----------  8 bits
   * - dePadIndex (0..28671) ------ 15 bits
   * - DsIndex (0..16819) --------- 15 bits
   * - solarIndex (0..623) -------- 10 bits
   * - elinkIndex (0..39) ---------  6 bits
   * - channel number (0..63) -----  6 bits
   */
  uint64_t mValue{sInvalidValue};

  ClassDefNV(ChannelCode, 1); // An identifier for a MCH channel
};

/** return a string representation */
std::string asString(const ChannelCode& cc);

inline bool operator==(const ChannelCode& a, const ChannelCode& b) { return a.value() == b.value(); }
inline bool operator!=(const ChannelCode& a, const ChannelCode& b) { return a.value() != b.value(); }
inline bool operator<(const ChannelCode& a, const ChannelCode& b) { return a.value() < b.value(); }
inline bool operator>(const ChannelCode& a, const ChannelCode& b) { return a.value() > b.value(); }
inline bool operator<=(const ChannelCode& a, const ChannelCode& b) { return a.value() <= b.value(); }
inline bool operator>=(const ChannelCode& a, const ChannelCode& b) { return a.value() >= b.value(); }

} // namespace o2::mch
#endif
