// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ZeroSuppressionLinkBased.h
/// \brief definitions to deal with the link based zero suppression format
/// \author Jens Wiechula

#ifndef ALICEO2_DATAFORMATSTPC_ZeroSuppressionLinkBased_H
#define ALICEO2_DATAFORMATSTPC_ZeroSuppressionLinkBased_H

#include <bitset>

namespace o2
{
namespace tpc
{
namespace zerosupp_link_based
{
static constexpr uint32_t DataWordSize = 16;                                ///< size of header word and data words
static constexpr uint32_t ChannelsPerWord = 10;                             ///< number of ADC values in one 128b word
static constexpr uint32_t DataBitSize = 12;                                 ///< number of bits of the data representation
static constexpr uint32_t SignificantBits = 2;                              ///< number of bits used for floating point precision
static constexpr float FloatConversion = 1.f / float(1 << SignificantBits); ///< conversion factor from integer representation to float

/// header definition of the zero suppressed link based data format
struct Header {
  union {
    uint64_t word0 = 0;             ///< lower 64 bits
    struct {                        ///
      uint64_t bitMaskLow : 64;     ///< lower bits of the 80 bit bitmask
    };                              ///
  };                                ///
                                    ///
  union {                           ///
    uint64_t word1 = 0;             ///< upper bits of the 80 bit bitmask
    struct {                        ///
      uint64_t bitMaskHigh : 16;    ///< higher bits of the 80 bit bitmask
      uint32_t bunchCrossing : 12;  ///< bunch crossing number
      uint32_t numWordsPayload : 8; ///< number of 128bit words with 12bit ADC values
      uint64_t zero : 28;           ///< not used
    };
  };

  std::bitset<80> getChannelBits()
  {
    return std::bitset<80>((std::bitset<80>(bitMaskHigh) << 64) | std::bitset<80>(bitMaskLow));
  }
};

struct Data {
  uint64_t adcValues[2]{}; ///< 128bit ADC values (max. 10x12bit)

  /// set ADC 'value' at position 'pos' (0-9)
  void setADCValue(uint32_t pos, uint64_t value)
  {
    if (pos < 5) {
      const uint64_t set = (value & uint64_t(0xFFF)) << (pos * 12);
      const uint64_t mask = (0xFFFFFFFFFFFFFFFF ^ (uint64_t(0xFFF) << (pos * 12)));
      adcValues[0] &= mask;
      adcValues[0] |= set;
    } else if (pos == 5) {
      const uint64_t set1 = (value & uint64_t(0xF)) << (pos * 12);
      const uint64_t set2 = (value >> 4) & uint64_t(0xFF);
      const uint64_t mask1 = 0x0FFFFFFFFFFFFFFF;
      const uint64_t mask2 = (0xFFFFFFFFFFFFFF00 ^ (uint64_t(0xFF) << (pos * 12)));
      adcValues[0] &= mask1;
      adcValues[0] |= set1;
      adcValues[1] &= mask2;
      adcValues[1] |= set2;
    } else {
      const uint64_t set = (value & uint64_t(0xFFF)) << ((pos * 12) + 8);
      const uint64_t mask = (0xFFFFFFFFFFF000FF ^ (uint64_t(0xFFF) << ((pos * 12) + 8)));
      adcValues[1] &= mask;
      adcValues[1] |= set;
    }
  }

  /// ADC value of channel at position 'pos' (0-9)
  uint32_t getADCValue(uint32_t pos)
  {
    if (pos < 5) {
      return (adcValues[0] >> (pos * 12)) & uint64_t(0xFFF);
    } else if (pos == 5) {
      return (adcValues[0] >> (pos * 12)) | ((adcValues[1] & uint64_t(0xFF)) << 4);
    } else {
      return (adcValues[1] >> (8 + pos * 12)) & uint64_t(0xFFF);
    }
  }
};

struct Container {
  Header header; ///< header data
  Data data[0];  ///< 128 bit words with 12bit ADC values

  /// return 12bit ADC value for a specific word in the data stream
  uint32_t getADCValue(uint32_t word) { return data[word / ChannelsPerWord].getADCValue(word % ChannelsPerWord); }

  /// return 12bit ADC value for a specific word in the data stream converted to float
  float getADCValueFloat(uint32_t word) { return float(getADCValue(word)) * FloatConversion; }

  /// set 12bit ADC value for a specific word in the data stream
  void setADCValue(uint32_t word, uint64_t value) { data[word / ChannelsPerWord].setADCValue(word % ChannelsPerWord, value); }

  /// return 12bit ADC value for a specific word in the data stream converted to float
  void setADCValueFloat(uint32_t word, float value) { data[word / ChannelsPerWord].setADCValue(word, uint64_t((value + FloatConversion / 2.f) / FloatConversion)); }

  /// get position of next container. Validity check to be done outside!
  Container* next()
  {
    return (Container*)(this + (header.numWordsPayload + 1) * DataWordSize);
  }
}; // namespace zerosupp_link_based

} // namespace zerosupp_link_based
} // namespace tpc
} // namespace o2

#endif
