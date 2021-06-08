// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_DATA_FORMATS_H
#define O2_MCH_RAW_DATA_FORMATS_H

#include <cstdint>
#include <iosfwd>

namespace o2::mch::raw
{

struct BareFormat {
};

struct UserLogicFormat {
};

struct ChargeSumMode {
  bool operator()() const { return true; }
};

struct SampleMode {
  bool operator()() const { return false; }
};

template <typename FORMAT>
struct isUserLogicFormat; // only defined (on purpose) for two types below

template <>
struct isUserLogicFormat<UserLogicFormat> {
  static constexpr bool value = true;
};

template <>
struct isUserLogicFormat<BareFormat> {
  static constexpr bool value = false;
};

template <typename CHARGESUM>
struct isChargeSumMode; // only defined (on purpose) for two types below

template <>
struct isChargeSumMode<ChargeSumMode> {
  static constexpr bool value = true;
};

template <>
struct isChargeSumMode<SampleMode> {
  static constexpr bool value = false;
};

using uint5_t = uint8_t;
using uint6_t = uint8_t;

using SampaChannelAddress = uint5_t; // 0..31 channel of a Sampa
using DualSampaChannelId = uint6_t;  // 0..63 channel of a *Dual* Sampa

using uint10_t = uint16_t;
using uint20_t = uint32_t;
using uint50_t = uint64_t;

// Format of 64 bits-words of the UserLogicFormat
template <int VERSION>
struct ULHeaderWord;

// initial UL format (2020)
template <>
struct ULHeaderWord<0> {
  union {
    uint64_t word;
    struct {
      uint64_t data : 50;
      uint64_t error : 2;
      uint64_t incomplete : 1;
      uint64_t dsID : 6;
      uint64_t linkID : 5;
    };
  };
};

// version 1 of UL format (2021)
// = as initial version with 1 bit less for linkID and 1 bit more for error
template <>
struct ULHeaderWord<1> {
  union {
    uint64_t word;
    struct {
      uint64_t data : 50;
      uint64_t error : 3;
      uint64_t incomplete : 1;
      uint64_t dsID : 6;
      uint64_t linkID : 4;
    };
  };
};

// structure of the FEEID field (16 bits) in the MCH Raw Data RDH
struct FEEID {
  union {
    uint16_t word;
    struct {
      uint16_t id : 8;
      uint16_t chargeSum : 1;
      uint16_t reserved : 3;
      uint16_t ulFormatVersion : 4;
    };
  };
};

template <typename CHARGESUM>
uint16_t extraFeeIdChargeSumMask();

template <int VERSION>
uint16_t extraFeeIdVersionMask();

template <typename FORMAT>
uint8_t linkRemapping(uint8_t linkID);

std::ostream& operator<<(std::ostream& os, const FEEID& f);
} // namespace o2::mch::raw

#endif
