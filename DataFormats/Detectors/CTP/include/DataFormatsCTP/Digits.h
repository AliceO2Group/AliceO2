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
static constexpr uint32_t CRULinkIDIntRec = 0;
static constexpr uint32_t NIntRecPayload = 48 + 12;
static constexpr uint32_t CRULinkIDClassRec = 1;
static constexpr uint32_t NClassPayload = 64 + 12;
static constexpr uint32_t NGBT = 80;
static constexpr std::uint32_t NumOfHBInTF = 256;
typedef std::bitset<NGBT> gbtword80_t;
//
static constexpr std::uint32_t CTP_NINPUTS = 46;    /// Max number of CTP inputs for all levels
static constexpr std::uint32_t CTP_NCLASSES = 64;   /// Number of classes in hardware
static constexpr std::uint32_t CTP_MAXTRIGINPPERDET = 5; /// Max number of LM/L0inputs per detector
/// Positions of CTP Detector inputs in CTPInputMask: first=offset, second=mask
/// For digits input position is  fixed
/// CTP hits are inputs. Digits are inputs collected from all detectors and CTP Class mask.
/// digits->raw: NO CTP Config to be used
/// raw->digits: NO CTP config to be used
/// Hits (CTP inputs) to CTP digits, i.e. inputs in correct position in mask nad CTP classes mask: CTP config to be used.
///
///
struct CTPDigit {
  o2::InteractionRecord intRecord;
  std::bitset<CTP_NINPUTS> CTPInputMask;
  std::bitset<CTP_NCLASSES> CTPClassMask;
  CTPDigit() = default;
  void printStream(std::ostream& stream) const;
  void setInputMask(gbtword80_t mask);
  void setClassMask(gbtword80_t mask);
  ClassDefNV(CTPDigit, 2);
};
struct CTPInputDigit {
  o2::InteractionRecord intRecord;
  std::bitset<CTP_MAXTRIGINPPERDET> inputsMask;
  o2::detectors::DetID::ID detector;
  CTPInputDigit() = default;
  ClassDefNV(CTPInputDigit, 1)
};
} // namespace ctp
} // namespace o2
#endif //_CTP_DIGITS_H
