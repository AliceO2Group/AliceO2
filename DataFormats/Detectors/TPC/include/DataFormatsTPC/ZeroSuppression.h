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

/// \file ZeroSuppression.h
/// \brief Definitions of TPC Zero Suppression Data Headers
/// \author David Rohr
#ifndef ALICEO2_DATAFORMATSTPC_ZEROSUPPRESSION_H
#define ALICEO2_DATAFORMATSTPC_ZEROSUPPRESSION_H
#ifndef GPUCA_GPUCODE_DEVICE
#include <cstdint>
#include <cstddef> // for size_t
#endif
#include "GPUCommonDef.h"
#include "GPUCommonRtypes.h"

namespace o2
{
namespace tpc
{

enum ZSVersion : unsigned char {
  ZSVersionRowBased10BitADC = 1,
  ZSVersionRowBased12BitADC = 2,
  ZSVersionLinkBasedWithMeta = 3,
  ZSVersionDenseLinkBased = 4,
  ZSVersionDenseLinkBasedV2 = 5
};

struct TPCZSHDR {
  static constexpr size_t TPC_ZS_PAGE_SIZE = 8192;
  static constexpr size_t TPC_MAX_SEQ_LEN = 138;
  static constexpr size_t TPC_MAX_ZS_ROW_IN_ENDPOINT = 9;
  static constexpr unsigned int MAX_DIGITS_IN_PAGE = (TPC_ZS_PAGE_SIZE - 64 - 6 - 4 - 3) * 8 / 10;
  static constexpr unsigned int TPC_ZS_NBITS_V1 = 10;
  static constexpr unsigned int TPC_ZS_NBITS_V2 = 12;

  unsigned char version;      // ZS format version:
                              // 1: original row-based format with 10-bit ADC values
                              // 2: original row-based format with 12-bit ADC values
                              // 3: improved link-based format with extra META header
                              // 4: dense link based
  unsigned char nTimeBinSpan; // Span of time bins in this raw page, i.e. last timeBin is <= timeOffset + nTimeBinSpan
  unsigned short cruID;       // CRU id
  unsigned short timeOffset;  // Time offset in BC after orbit in RDH
  unsigned short nADCsamples; // Total number of ADC samples in this raw page
};
struct TPCZSHDRV2 : public TPCZSHDR {
  static constexpr unsigned int TPC_ZS_NBITS_V34 = 12;
  static constexpr bool TIGHTLY_PACKED_V3 = false;
  static constexpr unsigned int SAMPLESPER64BIT = 64 / TPC_ZS_NBITS_V34; // 5 12-bit samples with 4 bit padding per 64 bit word for non-TIGHTLY_PACKED data
  static constexpr unsigned int TRIGGER_WORD_SIZE = 16;                  // trigger word size in bytes
  enum ZSFlags : unsigned char {
    TriggerWordPresent = 1,
    nTimeBinSpanBit8 = 2,
    payloadExtendsToNextPage = 4
  };

  unsigned short firstZSDataOffset; // zs Version   3: Offset (after the TPCZSHDRV2 header) in 128bit words to first ZS data (in between can be trigger words, etc.)
                                    // zs Version >=4: Offset (from beginning of page) in bytes of the first ZS data.
  unsigned short nTimebinHeaders;   // Number of timebin headers
  unsigned char flags;              // flag field (zs version 4 only): 0 = triggerWordPresent, 1 = bit 8 of nTimeBinSpan (i.e. nTimeBinSpan += 256)
  unsigned char reserved1;          // 16 reserved bits, header is 128 bit
  unsigned char reserved2;          // 8 reserved bits, header is 128 bit
  unsigned char magicWord;          // Magic word
};
struct TPCZSTBHDR {
  unsigned short rowMask;
  GPUd() unsigned short* rowAddr1() { return (unsigned short*)((unsigned char*)this + sizeof(*this)); }
  GPUd() const unsigned short* rowAddr1() const { return (unsigned short*)((unsigned char*)this + sizeof(*this)); }
};

struct ZeroSuppressedContainer { // Struct for the TPC zero suppressed data format
                                 // RDH 64 byte
                                 // 6 byte header for the zero suppressed format ; 8 bit version, 8 bit number of timebins, 16 bit CRU ID, 16 bit time offset
                                 // Time bin information
  unsigned long int rdh[8] = {}; //< 8 * 64 bit RDH (raw data header)
  TPCZSHDR hdr;                  // ZS header
};

/// Trigger word for dense link-base ZS
///
/// Trigger word is always 128bit and occurs always in the last page of a HBF before the meta header
struct TriggerWordDLBZS {
  static constexpr uint16_t MaxTriggerEntries = 8; ///< Maximum number of trigger information

  /// trigger types as in the ttype bits
  enum TriggerType : uint8_t {
    PhT = 1, ///< Physics Trigger
    PP = 2,  ///< Pre Pulse for calibration
    Cal = 4, ///< Laser (Calibration trigger)
  };
  uint16_t triggerEntries[MaxTriggerEntries] = {};

  uint16_t getTriggerBC(int entry = 0) const { return triggerEntries[entry] & 0xFFF; }
  uint16_t getTriggerType(int entry = 0) const { return (triggerEntries[entry] >> 12) & 0x7; }
  bool isValid(int entry = 0) const { return triggerEntries[entry] & 0x8000; }
};

/// Trigger info including the orbit
struct TriggerInfoDLBZS {
  TriggerWordDLBZS triggerWord{}; ///< trigger Word information
  uint32_t orbit{};               ///< orbit of the trigger word

  ClassDefNV(TriggerInfoDLBZS, 1);
};

} // namespace tpc
} // namespace o2
#endif
