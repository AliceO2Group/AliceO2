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

/// \file   MIDRaw/ELinkDataShaper.h
/// \brief  Properly formats and sets the absolute timestamp of the raw data
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   18 March 2021
#ifndef O2_MID_ELINKDATASHAPER_H
#define O2_MID_ELINKDATASHAPER_H

#include <cstdint>
#include <vector>
#include <functional>
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsMID/ROBoard.h"
#include "DataFormatsMID/ROFRecord.h"
#include "MIDRaw/ElectronicsDelay.h"
#include "MIDRaw/ELinkDecoder.h"

namespace o2
{
namespace mid
{
class ELinkDataShaper
{
 public:
  ELinkDataShaper(bool isDebugMode, bool isLoc, uint8_t uniqueId, const ElectronicsDelay& electronicsDelay);
  /// Main function to be executed when decoding is done
  inline void onDone(const ELinkDecoder& decoder, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs) { std::invoke(mOnDone, this, decoder, data, rofs); }

  void set(uint32_t orbit, uint32_t trigger);

 private:
  uint8_t mUniqueId = 0;                /// UniqueId
  ElectronicsDelay mElectronicsDelay{}; /// Delays in the electronics
  uint32_t mRDHOrbit = 0;               /// RDH orbit

  InteractionRecord mIR;          /// Interaction record
  InteractionRecord mExpectedFET; /// Expected FET clock
  int16_t mLocalToBCSelfTrig = 0; /// Local to BC for self-triggered events
  uint16_t mMaxBunches = 0;       /// Maximum number of bunches between orbits

  typedef void (ELinkDataShaper::*OnDoneFunction)(const ELinkDecoder&, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs);
  OnDoneFunction mOnDone{&ELinkDataShaper::onDoneLoc}; ///! Processes the board

  void onDoneLoc(const ELinkDecoder& decoder, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs);
  void onDoneLocDebug(const ELinkDecoder& decoder, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs);
  void onDoneReg(const ELinkDecoder&, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs){}; /// Dummy function
  void onDoneRegDebug(const ELinkDecoder& decoder, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs);

  void addLoc(const ELinkDecoder& decoder, EventType eventType, InteractionRecord ir, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs);
  bool checkLoc(const ELinkDecoder& decoder);
  EventType processCalibrationTrigger(const InteractionRecord& ir);
  void processHBTrigger(uint16_t localClock, uint8_t triggerWord);
  EventType processSelfTriggered(InteractionRecord& ir);
  bool processTrigger(const ELinkDecoder& decoder, EventType& eventType, InteractionRecord& ir);
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_ELINKDATASHAPER_H */
