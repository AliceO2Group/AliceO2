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

static constexpr uint32_t DataWordSizeBits = 128;                   ///< size of header word and data words in bits
static constexpr uint32_t DataWordSizeBytes = DataWordSizeBits / 8; ///< size of header word and data words in bytes

/// header definition of the zero suppressed link based data format
struct Header {
  static constexpr uint32_t MagicWord = 0xFC000000;

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
      uint32_t numWordsPayload : 4; ///< number of 128bit words with 12bit ADC values
      uint32_t magicWord : 32;      ///< not used
    };
  };

  std::bitset<80> getChannelBits() const
  {
    return std::bitset<80>((std::bitset<80>(bitMaskHigh) << 64) | std::bitset<80>(bitMaskLow));
  }

  bool hasCorrectMagicWord() const { return magicWord == MagicWord; }
};

/// empty header for

/// ADC data container
///
/// In case of zero suppressed data, the ADC values are stored with 12 bit
/// 10bit + 2bit precision
/// In case of decoded raw data, the pure 10 bit ADC values are stored
///
/// The data is packed in 128 bit words, or 2x64 bit. Each 64 bit word has 4 bit
/// padding.
/// So it is either 2 x ((5 x 12 bit) + 4 bit padding), or
///                 2 x ((6 x 10 bit) + 4 bit padding)
template <uint32_t DataBitSizeT = 12, uint32_t SignificantBitsT = 2>
struct Data {
  static constexpr uint32_t ChannelsPerWord = DataWordSizeBits / DataBitSizeT; ///< number of ADC values in one 128b word
  static constexpr uint32_t ChannelsPerHalfWord = ChannelsPerWord / 2;         ///< number of ADC values in one 64b word
  static constexpr uint32_t DataBitSize = DataBitSizeT;                        ///< number of bits of the data representation
  static constexpr uint32_t SignificantBits = SignificantBitsT;                ///< number of bits used for floating point precision
  static constexpr uint64_t BitMask = ((uint64_t(1) << DataBitSize) - 1);      ///< mask for bits
  static constexpr float FloatConversion = 1.f / float(1 << SignificantBits);  ///< conversion factor from integer representation to float

  uint64_t adcValues[2]{}; ///< 128bit ADC values (max. 10x12bit)

  /// set ADC 'value' at position 'pos' (0-9)
  void setADCValue(uint32_t pos, uint64_t value)
  {
    const uint32_t word = pos / ChannelsPerHalfWord;
    const uint32_t posInWord = pos % ChannelsPerHalfWord;

    const uint64_t set = (value & BitMask) << (posInWord * DataBitSize);
    const uint64_t mask = (0xFFFFFFFFFFFFFFFF ^ (BitMask << (posInWord * DataBitSize)));

    adcValues[word] &= mask;
    adcValues[word] |= set;
  }

  /// set ADC value from float
  void setADCValueFloat(uint32_t pos, float value)
  {
    setADCValue(pos, uint64_t((value + 0.5f * FloatConversion) / FloatConversion));
  }

  /// get ADC value of channel at position 'pos' (0-9)
  uint32_t getADCValue(uint32_t pos) const
  {
    const uint32_t word = pos / ChannelsPerHalfWord;
    const uint32_t posInWord = pos % ChannelsPerHalfWord;

    return (adcValues[word] >> (posInWord * DataBitSize)) & BitMask;
  }

  /// get ADC value in float
  float getADCValueFloat(uint32_t pos) const
  {
    return float(getADCValue(pos)) * FloatConversion;
  }

  /// reset all ADC values
  void reset()
  {
    adcValues[0] = 0;
    adcValues[1] = 0;
  }
};

template <uint32_t DataBitSizeT, uint32_t SignificantBitsT, bool HasHeaderT>
struct ContainerT;

template <uint32_t DataBitSizeT, uint32_t SignificantBitsT>
struct ContainerT<DataBitSizeT, SignificantBitsT, true> {
  Header header;                                ///< header data
  Data<DataBitSizeT, SignificantBitsT> data[0]; ///< 128 bit words with 12bit ADC values

  /// bunch crossing number
  uint32_t getBunchCrossing() const { return header.bunchCrossing; }

  /// number of data words without the header
  uint32_t getDataWords() const { return header.numWordsPayload; }

  /// number of data words including the header
  uint32_t getTotalWords() const { return header.numWordsPayload + 1; }

  /// channel bitmask
  std::bitset<80> getChannelBits() const { return header.getChannelBits(); }
};

template <uint32_t DataBitSizeT, uint32_t SignificantBitsT>
struct ContainerT<DataBitSizeT, SignificantBitsT, false> {
  Data<DataBitSizeT, SignificantBitsT> data[0]; ///< 128 bit words with 12bit ADC values

  /// bunch crossing number
  uint32_t getBunchCrossing() const { return 0; }

  /// number of data words without the header
  uint32_t getDataWords() const { return 7; }

  /// number of data words including the header, which is not present in case of 10bit decoded data format
  uint32_t getTotalWords() const { return 7; }

  /// channel bitmask
  std::bitset<80> getChannelBits() const { return std::bitset<80>().set(); }
};

/// Container for decoded data, either zero suppressed or pure raw data
///
/// In case of pure raw data, no header is needed, since all 80 channels will be filled
template <uint32_t DataBitSizeT = 12, uint32_t SignificantBitsT = 2, bool HasHeaderT = true>
struct Container {
  static constexpr uint32_t ChannelsPerWord = DataWordSizeBits / DataBitSizeT; ///< number of ADC values in one 128b word

  ContainerT<DataBitSizeT, SignificantBitsT, HasHeaderT> cont; ///< Templated data container

  /// return 12bit ADC value for a specific word in the data stream
  uint32_t getADCValue(uint32_t word) { return cont.data[word / ChannelsPerWord].getADCValue(word % ChannelsPerWord); }

  /// return 12bit ADC value for a specific word in the data stream converted to float
  float getADCValueFloat(uint32_t word) { return cont.data[word / ChannelsPerWord].getADCValueFloat(word % ChannelsPerWord); }

  /// set 12bit ADC value for a specific word in the data stream
  void setADCValue(uint32_t word, uint64_t value) { cont.data[word / ChannelsPerWord].setADCValue(word % ChannelsPerWord, value); }

  /// return 12bit ADC value for a specific word in the data stream converted to float
  void setADCValueFloat(uint32_t word, float value) { cont.data[word / ChannelsPerWord].setADCValueFloat(word % ChannelsPerWord, value); }

  /// reset all ADC values
  void reset()
  {
    for (int i = 0; i < cont.getDataWords(); ++i) {
      cont.data[i].reset();
    }
  }

  uint32_t getBunchCrossing() const { return cont.getBunchCrossing(); }
  uint32_t getDataWords() const { return cont.getDataWords(); }
  uint32_t getTotalWords() const { return cont.getTotalWords(); }
  std::bitset<80> getChannelBits() const { return cont.getChannelBits(); }

  /// total size in bytes
  size_t getTotalSizeBytes() const { return getTotalWords() * DataWordSizeBytes; }

  /// get position of next container. Validity check to be done outside!
  Container* next() const
  {
    return (Container*)((const char*)this + getTotalWords() * DataWordSizeBytes);
  }

}; // namespace zerosupp_link_based

using ContainerZS = Container<>;
using ContainerDecoded = Container<10, 0, false>;

} // namespace zerosupp_link_based
} // namespace tpc
} // namespace o2

#endif
