// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file IDC.h
/// \brief Integrated digital currents data format definition
/// @author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de
/// The data is sent by the CRU as 256bit words. The IDC data layout is as follows:
/// Header     [            256 bits                  ]
/// Channel-00 [L9][L8][L7][L6][L5][L4][L3][L2][L1][L0]
/// ...
/// Channel-79 [L9][L8][L7][L6][L5][L4][L3][L2][L1][L0]
///
/// Where [Lx] is a 25bit value for Channel yy link x

#ifndef ALICEO2_DATAFORMATSTPC_IDC_H
#define ALICEO2_DATAFORMATSTPC_IDC_H

#include <bitset>

namespace o2::tpc::idc
{
static constexpr uint32_t Links = 10;                                           ///< maximum number of links
static constexpr uint32_t Channels = 80;                                        ///< number of channels
static constexpr uint32_t DataWordSizeBits = 256;                               ///< size of header word and data words in bits
static constexpr uint32_t DataWordSizeBytes = DataWordSizeBits / 8;             ///< size of header word and data words in bytes
static constexpr uint32_t IDCvalueBits = 25;                                    ///< number of bits used for one IDC value
static constexpr uint32_t IDCvalueBitsMask = (uint32_t(1) << IDCvalueBits) - 1; ///< bitmask for 25 bit word

/// header definition of the IDCs
/// The header is a 256 bit word
struct Header {
  static constexpr uint32_t MagicWord = 0xDC;
  union {
    uint64_t word0 = 0;              ///< bits 0 - 63
    struct {                         ///
      uint32_t version : 8;          ///< lower bits of the 80 bit bitmask
      uint32_t packetID : 8;         ///< packet id
      uint32_t errorCode : 8;        ///< errors
      uint32_t magicWord : 8;        ///< magic word
      uint32_t heartbeatOrbit : 32;  ///< heart beat orbit of the IDC value
    };                               ///
  };                                 ///
                                     ///
  union {                            ///
    uint64_t word1 = 0;              ///< bits 64 - 127
    struct {                         ///
      uint32_t heartbeatBC : 16;     ///< BC id of IDC value
      uint32_t integrationTime : 16; ///< integration time used for the IDCs
      uint32_t linkMask : 16;        ///< mask of active links
      uint32_t unused1 : 16;         ///
    };                               ///
  };                                 ///
                                     ///
  union {                            ///
    uint64_t word2 = 0;              ///< bits 128 - 191
    struct {                         ///
      uint64_t unused2 : 64;         ///< lower bits of the 80 bit bitmask
    };                               ///
  };                                 ///
                                     ///
  union {                            ///
    uint64_t word3 = 0;              ///< bits 192 - 255
    struct {                         ///
      uint64_t unused3 : 64;         ///< lower bits of the 80 bit bitmask
    };                               ///
  };                                 ///
                                     ///
};

/// IDC single channel data container
/// TODO: verify that pointer arithmetics does not run into alignment issues
///       might require different logic
struct Data {
  uint8_t dataWords[DataWordSizeBytes] = {0}; ///< 25bit ADC values

  uint32_t getLinkValue(const uint32_t link) const
  {
    const auto valPtr = dataWords;
    const uint32_t offset = link * IDCvalueBits;
    const uint32_t selectedWord = offset / 8;
    const uint32_t requiredShift = offset % 8;
    const uint32_t value = (*(uint32_t*)(dataWords + selectedWord)) >> requiredShift;
    return value & IDCvalueBitsMask;
  }

  void setLinkValue(const uint32_t link, const uint32_t value)
  {
    const uint32_t offset = link * IDCvalueBits;
    const uint32_t selectedWord = offset / 8;
    const uint32_t requiredShift = offset % 8;
    auto dataWrite = (uint64_t*)&dataWords[selectedWord];
    *dataWrite = (value & IDCvalueBitsMask) << requiredShift;
  }
};

/// IDC full data container
struct Container {
  Header header;              ///< IDC data header
  Data channelData[Channels]; ///< data values for all channels in each link

  uint32_t getChannelValue(const uint32_t link, const uint32_t channel) const
  {
    return channelData[channel].getLinkValue(link);
  }

  void setChannelValue(const uint32_t link, const uint32_t channel, uint32_t value)
  {
    channelData[channel].setLinkValue(link, value);
  }
};
} // namespace o2::tpc::idc
#endif
