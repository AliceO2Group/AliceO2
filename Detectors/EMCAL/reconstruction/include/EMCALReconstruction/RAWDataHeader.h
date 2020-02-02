// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_EMCAL_RAwDATAHEADER_H
#define ALICEO2_EMCAL_RAwDATAHEADER_H

#include <iosfwd>
#include <cstdint>
#include "Rtypes.h"

namespace o2
{
namespace emcal
{
struct RAWDataHeader {
  union {
    uint32_t word0 = 0xFFFFFFFF; // size of the raw data in bytes
  };

  union {
    uint32_t word1 = 3 << 24; // bunch crossing, L1 trigger message and format version
    struct {
      uint32_t triggerBC : 14;       ///< bunch crossing [0-13]
      uint32_t triggermessageL1 : 8; ///< L1 trigger message [14-21]
      uint32_t zero11 : 2;           ///< Unassigned [22-23]
      uint32_t version : 8;          ///< Version [24-31]
    };
  };

  union {
    uint32_t word2 = 0; //< Size and offset
    struct {
      uint32_t offsetToNext : 16; ///< offset [0-15]
      uint32_t memorySize : 16;   ///< size [16-31]
    };
  };

  union {
    uint32_t word3 = 0; ///< Number of packets and linkID
    struct {
      uint8_t linkID : 8;        ///< Link ID  [0-7]
      uint8_t packetCounter : 8; ///< Number of packets  [8-15]
      uint16_t zero31 : 16;      ///< Unassigned [16-31]
    };
  };

  union {
    uint32_t word4 = 0x10000; // status & error bits and mini event ID
    struct {
      uint32_t triggerOrbit : 12; ///< mini event ID [0-11]
      uint32_t status : 16;       ///< status & error bits [12-27]
      uint32_t zero41 : 4;        ///< Unassigned [28-31]
    };
  };

  union {
    uint32_t word5; ///< First word of the tirgger types
    struct {
      uint32_t triggerType : 32; ///< low trigger types [0-49]
    };
  };

  union {
    uint32_t word6; ///< Second word of the trigger types
    struct {
      uint32_t triggerTypesHigh : 18;   ///< Second part of the trigger types [0-17]
      uint32_t triggerTypesNext50 : 14; ///< First part of the trigger types next 50 [18-31]
    };
  };

  union {
    uint32_t word7; ///< Third word of the trigger types
    struct {
      uint32_t triggerTypesNext50Middle : 32; ///< Second part of the trigger types next 50
    };
  };

  union {
    uint32_t word8; ///< Fourth word of the trigger types
    struct {
      uint32_t triggerTypesNext50High : 4; ///< Third part of the trigger types next 50 [0-3]
      uint32_t zero81 : 24;                ///< Unassigned [4-27]
      uint32_t roi : 4;                    ///< First part of the roi [28-31]
    };
  };

  union {
    uint32_t word9; ///< Second word of the roi
    struct {
      uint32_t roiHigh : 32; ///< Second part of the roi
    };
  };
};

std::istream& operator>>(std::istream& in, o2::emcal::RAWDataHeader& header);
std::ostream& operator<<(std::ostream& out, const o2::emcal::RAWDataHeader& header);

} // namespace emcal

} // namespace o2

#endif // _O2_EMCAL_RAwDATAHEADER_H__