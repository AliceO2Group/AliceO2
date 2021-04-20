// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  ELinkDataShaper(bool isDebugMode, bool isLoc, uint8_t uniqueId);
  /// Main function to be executed when decoding is done
  inline void onDone(const ELinkDecoder& decoder, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs) { std::invoke(mOnDone, this, decoder, data, rofs); }

  void set(uint32_t orbit);

  /// Sets the delay in the electronics
  void setElectronicsDelay(const ElectronicsDelay& electronicsDelay) { mElectronicsDelay = electronicsDelay; }

 private:
  uint8_t mUniqueId{0};                 /// UniqueId
  uint32_t mRDHOrbit{0};                /// RDH orbit
  bool mReceivedCalibration{false};     /// Flag to indicate if the  calibration trigger was received
  ElectronicsDelay mElectronicsDelay{}; /// Delays in the electronics

  InteractionRecord mIR{};      /// Interaction record
  uint16_t mExpectedFETClock{}; /// Expected FET clock
  uint16_t mLastClock{};        /// Last clock

  typedef void (ELinkDataShaper::*OnDoneFunction)(const ELinkDecoder&, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs);
  OnDoneFunction mOnDone{&ELinkDataShaper::onDoneLoc}; ///! Processes the board

  void onDoneLoc(const ELinkDecoder& decoder, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs);
  void onDoneLocDebug(const ELinkDecoder& decoder, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs);
  void onDoneReg(const ELinkDecoder&, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs){}; /// Dummy function
  void onDoneRegDebug(const ELinkDecoder& decoder, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs);

  void addLoc(const ELinkDecoder& decoder, EventType eventType, uint16_t correctedClock, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs);
  bool checkLoc(const ELinkDecoder& decoder);
  EventType processCalibrationTrigger(uint16_t localClock);
  void processOrbitTrigger(uint16_t localClock, uint8_t triggerWord);
  EventType processSelfTriggered(uint16_t localClock, uint16_t& correctedClock);
  bool processTrigger(const ELinkDecoder& decoder, EventType& eventType, uint16_t& correctedClock);
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_ELINKDATASHAPER_H */
