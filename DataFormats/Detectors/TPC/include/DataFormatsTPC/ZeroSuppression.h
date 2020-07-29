// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

namespace o2
{
namespace tpc
{

struct TPCZSHDR {
  static constexpr size_t TPC_ZS_PAGE_SIZE = 8192;
  static constexpr size_t TPC_MAX_SEQ_LEN = 138;
  static constexpr size_t TPC_MAX_ZS_ROW_IN_ENDPOINT = 9;
  static constexpr unsigned int MAX_DIGITS_IN_PAGE = (TPC_ZS_PAGE_SIZE - 64 - 6 - 4 - 3) * 8 / 10;
  static constexpr unsigned int TPC_ZS_NBITS_V1 = 10;
  static constexpr unsigned int TPC_ZS_NBITS_V2 = 12;

  unsigned char version;
  unsigned char nTimeBins;
  unsigned short cruID;
  unsigned short timeOffset;
  unsigned short nADCsamples;
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
  TPCZSTBHDR tbhdr;              // ZS timebin header
};

} // namespace tpc
} // namespace o2
#endif
