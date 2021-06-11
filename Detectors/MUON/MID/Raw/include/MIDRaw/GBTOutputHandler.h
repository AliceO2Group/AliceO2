// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDRaw/GBTOutputHandler.h
/// \brief  MID GBT decoder output handler
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   14 April 2020
#ifndef O2_MID_GBTOUTPUTHANDLER_H
#define O2_MID_GBTOUTPUTHANDLER_H

#include <cstdint>
#include <array>
#include <vector>
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsMID/ROFRecord.h"
#include "MIDRaw/CrateParameters.h"
#include "MIDRaw/ElectronicsDelay.h"
#include "MIDRaw/ELinkDecoder.h"
#include "DataFormatsMID/ROBoard.h"

namespace o2
{
namespace mid
{
class GBTOutputHandler
{
 public:
  /// Sets the FEE Id
  void setGBTUniqueId(uint16_t feeId) { mFeeId = feeId; }

  void set(uint32_t orbit, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs);

  void onDoneLoc(size_t ilink, const ELinkDecoder& decoder);
  void onDoneLocDebug(size_t ilink, const ELinkDecoder& decoder);
  void onDoneReg(size_t, const ELinkDecoder&){}; /// Dummy function
  void onDoneRegDebug(size_t ilink, const ELinkDecoder& decoder);

  /// Sets the delay in the electronics
  void setElectronicsDelay(const ElectronicsDelay& electronicsDelay) { mElectronicsDelay = electronicsDelay; }

 private:
  std::vector<ROBoard>* mData{nullptr};         ///! Vector of output data. Not owner
  std::vector<ROFRecord>* mROFRecords{nullptr}; /// List of ROF records. Not owner
  uint16_t mFeeId{0};                           /// FEE ID
  uint32_t mOrbit{};                            /// RDH orbit
  uint16_t mReceivedCalibration{0};             /// Word with one bit per e-link indicating if the calibration trigger was received by the e-link
  ElectronicsDelay mElectronicsDelay{};         /// Delays in the electronics

  std::array<InteractionRecord, crateparams::sNELinksPerGBT> mIRs{};     /// Interaction records per link
  std::array<uint16_t, crateparams::sNELinksPerGBT> mExpectedFETClock{}; /// Expected FET clock
  std::array<uint16_t, crateparams::sNELinksPerGBT> mLastClock{};        /// Last clock per link

  void addLoc(size_t ilink, const ELinkDecoder& decoder, EventType eventType, uint16_t correctedClock);
  bool checkLoc(size_t ilink, const ELinkDecoder& decoder);
  EventType processCalibrationTrigger(size_t ilink, uint16_t localClock);
  void processOrbitTrigger(size_t ilink, uint16_t localClock, uint8_t triggerWord);
  EventType processSelfTriggered(size_t ilink, uint16_t localClock, uint16_t& correctedClock);
  bool processTrigger(size_t ilink, const ELinkDecoder& decoder, EventType& eventType, uint16_t& correctedClock);
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_GBTOUTPUTHANDLER_H */
