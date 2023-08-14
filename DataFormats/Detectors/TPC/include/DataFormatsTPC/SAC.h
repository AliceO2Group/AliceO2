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

/// \file SAC.h
/// \brief Sampled Analogue Currents (SACs) data format definitions
/// @author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef ALICEO2_DATAFORMATSTPC_SAC_H
#define ALICEO2_DATAFORMATSTPC_SAC_H

#include <cstdint>
#include <string_view>

namespace o2::tpc::sac
{

static constexpr uint32_t FEsPerInstance = 9;

/// 256bit CRU header word of SACs
struct headerDef {
  static constexpr uint32_t MagicWord = 0xabcd; ///< magic word

  union {
    uint64_t word0 = 0;            ///< bits 0 - 63
    struct {                       ///
      uint32_t version : 8;        ///< header version number
      uint32_t instance : 4;       ///< 0: TPC A (slave), 1: TPC C (master), 2/3: TRD
      uint32_t empty0 : 8;         ///< not used
      uint32_t bunchCrossing : 12; ///< bunch crossing when SAC packet was received
      uint32_t orbit : 32;         ///< orbit when SAC packet was received
    };                             ///
  };                               ///
                                   ///
  union {                          ///
    uint64_t word1 = 0;            ///< bits 64 - 127
    struct {                       ///<
      uint32_t pktCount : 16;      ///< internal packet counter, should always increase by one
      uint32_t empty1_0 : 16;      ///< not used
      uint32_t empty1_1;           ///< not used
    };                             ///
  };                               ///
                                   ///
  union {                          ///
    uint64_t word2 = 0;            ///< bits 128 - 191
  };                               ///
                                   ///
  union {                          ///
    uint64_t word3 = 0;            ///< bits 192 - 255
    struct {                       ///
      uint32_t empty3_0;           ///<
      uint32_t empty3_1 : 16;      ///<
      uint32_t magicWord : 16;     ///< magic word should always correspond to MagicWord
    };                             ///
  };                               ///

  bool check() const { return magicWord == MagicWord; }
};

struct dataDef {
  static constexpr uint32_t HeaderWord = 0xdeadbeef;
  static constexpr uint32_t TrailerWord = 0xbeefdead;
  static constexpr uint32_t PacketSize = 1024;
  static constexpr uint32_t DataSize = 1000;

  uint32_t header = 0;       ///< header magic word, should always be HeaderWord
                             ///
  union {                    ///
    uint32_t sizeFE = 0;     ///<
    struct {                 ///
      uint32_t feid : 8;     ///< front end ID (card number in rack): 1 - 9
      uint32_t pktSize : 24; ///< total size of the packet in bytes, should always be 1024
    };                       ///
  };                         ///
                             ///
  uint32_t pktNumber = 0;    ///< packet number of this front end card. Should always increase by 1
  uint32_t timeStamp = 0;    ///< time stamp
  char dataWords[DataSize];  ///< ASCI encoded SAC data
  uint32_t crc32 = 0;        ///< CRC32 checksum
  uint32_t trailer = 0;      ///< trailer magic word, should always be TrailerWord

  bool check() const
  {
    return (header == HeaderWord) && (trailer == TrailerWord) && (pktSize == PacketSize);
  }
};

struct packet {
  headerDef header;
  dataDef data;

  uint32_t getInstance() const
  {
    return header.instance;
  }

  /// return the global front end card index
  /// 0 - 9: A-Side
  /// 10 - 17: C-Side
  uint32_t getFEIndex() const
  {
    return data.feid - 1 + header.instance * FEsPerInstance;
  }

  std::string_view getDataWords()
  {
    return std::string_view(data.dataWords, dataDef::DataSize);
  }

  bool check() const
  {
    return header.check() && data.check();
  }

  int getCheckMask() const
  {
    return (header.check() << 1) | data.check();
  }
};

} // namespace o2::tpc::sac

#endif
