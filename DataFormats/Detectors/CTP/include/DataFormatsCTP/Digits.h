// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Digits.h
/// \brief definition of CTPDigit, CTPInputDigit
/// \author Roman Lietava

#ifndef _CTP_DIGITS_H_
#define _CTP_DIGITS_H_
#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsFT0/Digit.h"
#include <bitset>
#include <iosfwd>

namespace o2
{
namespace ctp
{
/// CTP related constants
static constexpr std::uint32_t NumOfHBInTF = 256;
static constexpr std::uint32_t CTP_NINPUTS = 46;    /// Max number of CTP inputs for all levels
static constexpr std::uint32_t CTP_NCLASSES = 64;   /// Number of classes in hardware
static constexpr std::uint32_t CTP_MAXL0PERDET = 5; /// Max number of LM/L0inputs per detector
/// Positions of CTP Detector inputs in CTPInputMask
static constexpr std::pair<uint32_t, std::bitset<CTP_MAXL0PERDET>> CTP_INPUTMASK_FV0(0, 0x1f);
static constexpr std::pair<uint32_t, std::bitset<CTP_MAXL0PERDET>> CTP_INPUTMASK_FT0(5, 0x1f);
///
struct CTPDigit {
  o2::InteractionRecord intRecord;
  std::bitset<CTP_NINPUTS> CTPInputMask;
  std::bitset<CTP_NCLASSES> CTPClassMask;
  CTPDigit() = default;
  void printStream(std::ostream& stream) const;
  ClassDefNV(CTPDigit, 1);
};
struct CTPInputDigit {
  o2::InteractionRecord intRecord;
  std::bitset<CTP_MAXL0PERDET> inputsMask;
  o2::detectors::DetID::ID detector;
  CTPInputDigit() = default;
  ClassDefNV(CTPInputDigit, 1)
};
} // namespace ctp
} // namespace o2
#endif //_CTP_DIGITS_H
