// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef _FV0_CHANNEL_DATA_H_
#define _FV0_CHANNEL_DATA_H_

#include <Framework/Logger.h>
#include <array>
#include <Rtypes.h>
#include <tuple>
/// \file ChannelData.h
/// \brief Container class to store time and charge values of single FV0 channel

namespace o2
{
namespace fv0
{

struct ChannelData {
  static constexpr char sChannelNameDPL[] = "DIGITSCH";
  static constexpr char sDigitName[] = "ChannelData";
  static constexpr char sDigitBranchName[] = "FV0DigitCh";
  Short_t pmtNumber = -1; // PhotoMultiplier number (0 to 47)
  Short_t time = -1;      // [ns] Time associated with rising edge of the singal in a given channel
  Short_t chargeAdc = -1; // ADC sample as present in raw data

  ChannelData() = default;
  ChannelData(Short_t iPmt, Float_t t, Short_t charge)
  {
    pmtNumber = iPmt;
    time = t;
    chargeAdc = charge;
  }
  Short_t getChannelID() const { return pmtNumber; }
  void print() const;
  void printLog() const;
  bool operator==(ChannelData const& other) const
  {
    return std::tie(pmtNumber, time, chargeAdc) == std::tie(other.pmtNumber, other.time, other.chargeAdc);
  }
  ClassDefNV(ChannelData, 1);
};
} // namespace fv0
} // namespace o2

#endif
