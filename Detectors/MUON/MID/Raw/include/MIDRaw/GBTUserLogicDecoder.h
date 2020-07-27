// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDRaw/GBTUserLogicDecoder.h
/// \brief  MID GBT decoder with user logic zero suppression
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   15 April 2020
#ifndef O2_MID_GBTUSERLOGICDECODER_H
#define O2_MID_GBTUSERLOGICDECODER_H

#include <cstdint>
#include <vector>
#include <gsl/gsl>
#include "DetectorsRaw/RDHUtils.h"
#include "DataFormatsMID/ROFRecord.h"
#include "MIDRaw/ElectronicsDelay.h"
#include "MIDRaw/ELinkDecoder.h"
#include "MIDRaw/GBTOutputHandler.h"
#include "MIDRaw/LocalBoardRO.h"

namespace o2
{
namespace mid
{
class GBTUserLogicDecoder
{
 public:
  void init(uint16_t feeId, bool isDebugMode = false);
  void process(gsl::span<const uint8_t> bytes, uint16_t bc, uint32_t orbit, uint16_t pageCnt);
  template <typename RDH>
  void process(gsl::span<const uint8_t> bytes, const RDH& rdh)
  {
    process(bytes, o2::raw::RDHUtils::getHeartBeatBC(rdh), o2::raw::RDHUtils::getHeartBeatOrbit(rdh), o2::raw::RDHUtils::getPageCounter(rdh));
  }
  /// Gets the vector of data
  const std::vector<LocalBoardRO>& getData() const { return mOutputHandler.getData(); }

  /// Gets the vector of data RO frame records
  const std::vector<ROFRecord>& getROFRecords() const { return mOutputHandler.getROFRecords(); }

  /// Checks that the link has finished reading
  bool isComplete() const { return mELinkDecoder.isComplete(); }

  /// Clears the decoded data
  void clear() { mOutputHandler.clear(); }

  /// Sets the delay in the electronics
  void setElectronicsDelay(const ElectronicsDelay& electronicsDelay) { mOutputHandler.setElectronicsDelay(electronicsDelay); }

 private:
  GBTOutputHandler mOutputHandler{}; /// GBT output handler

  ELinkDecoder mELinkDecoder{}; /// E-link decoder

  typedef void (GBTOutputHandler::*OnDoneFunction)(size_t, const ELinkDecoder&);

  OnDoneFunction mOnDoneLoc{&GBTOutputHandler::onDoneLoc}; ///! Processes the local board
  OnDoneFunction mOnDoneReg{&GBTOutputHandler::onDoneReg}; ///! Processes the regional board
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_GBTUSERLOGICDECODER_H */
