// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDRaw/GBTBareDecoder.h
/// \brief  MID GBT decoder without user logic
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   12 March 2020
#ifndef O2_MID_GBTBAREDECODER_H
#define O2_MID_GBTBAREDECODER_H

#include <cstdint>
#include <array>
#include <vector>
#include <gsl/gsl>
#include "DataFormatsMID/ROFRecord.h"
#include "MIDRaw/CrateParameters.h"
#include "MIDRaw/ElectronicsDelay.h"
#include "MIDRaw/ELinkDecoder.h"
#include "MIDRaw/GBTOutputHandler.h"
#include "MIDRaw/LocalBoardRO.h"

namespace o2
{
namespace mid
{
class GBTBareDecoder
{
 public:
  void init(uint16_t feeId, uint8_t mask, bool isDebugMode = false);
  void process(gsl::span<const uint8_t> bytes, uint16_t bc, uint32_t orbit, uint16_t pageCnt);
  /// Gets the vector of data
  const std::vector<LocalBoardRO>& getData() const { return mOutputHandler.getData(); }

  /// Gets the vector of data RO frame records
  const std::vector<ROFRecord>& getROFRecords() const { return mOutputHandler.getROFRecords(); }

  bool isComplete() const;

  /// Clears the decoded data
  void clear() { mOutputHandler.clear(); }

  /// Sets the delay in the electronics
  void setElectronicsDelay(const ElectronicsDelay& electronicsDelay) { mOutputHandler.setElectronicsDelay(electronicsDelay); }

 private:
  GBTOutputHandler mOutputHandler{}; /// GBT output handler
  uint8_t mMask{0xFF};               /// GBT mask
  uint16_t mIsFeeding{0};            /// Flag to check if the e-link is feeding

  std::array<ELinkDecoder, crateparams::sNELinksPerGBT> mELinkDecoders{}; /// E-link decoders

  // Here we are using a function pointer instead of a std::function because it is faster.
  // The std::function adds an overhead at each function call,
  // which results in a considerable slowing done of the code if the function is executed often
  typedef void (GBTOutputHandler::*OnDoneFunction)(size_t, const ELinkDecoder&);
  typedef void (GBTBareDecoder::*ProcessFunction)(size_t, uint8_t);

  OnDoneFunction mOnDoneLoc{&GBTOutputHandler::onDoneLoc};  ///! Processes the local board
  ProcessFunction mProcessReg{&GBTBareDecoder::processReg}; ///! Processes the regional board

  void processLoc(size_t ilink, uint8_t byte);
  void processReg(size_t, uint8_t){}; /// Dummy function. We usually do not process the regional cards, except when we are debugging the code
  void processRegDebug(size_t ilink, uint8_t byte);
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_GBTBAREDECODER_H */
