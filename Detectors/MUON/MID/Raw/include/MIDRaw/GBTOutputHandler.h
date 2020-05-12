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
#include "MIDRaw/ELinkDecoder.h"
#include "MIDRaw/LocalBoardRO.h"

namespace o2
{
namespace mid
{
class GBTOutputHandler
{
 public:
  /// Sets the FEE Id
  void setFeeId(uint16_t feeId) { mFeeId = feeId; }

  void setIR(uint16_t bc, uint32_t orbit, int pageCnt);

  void onDoneLoc(std::size_t ilink, const ELinkDecoder& decoder);
  void onDoneLocDebug(std::size_t ilink, const ELinkDecoder& decoder);
  void onDoneReg(std::size_t, const ELinkDecoder&){}; /// Dummy function
  void onDoneRegDebug(std::size_t ilink, const ELinkDecoder& decoder);

  /// Gets the vector of data
  const std::vector<LocalBoardRO>& getData() const { return mData; }

  /// Gets the vector of data RO frame records
  const std::vector<ROFRecord>& getROFRecords() const { return mROFRecords; }

  void clear();

 private:
  std::vector<LocalBoardRO> mData{};    /// Vector of output data
  std::vector<ROFRecord> mROFRecords{}; /// List of ROF records
  uint16_t mFeeId{0};                   /// FEE ID
  InteractionRecord mIRFirstPage{};     /// Interaction record of the first page

  std::array<InteractionRecord, crateparams::sNELinksPerGBT> mIRs{}; /// Interaction records per link
  std::array<uint16_t, crateparams::sNELinksPerGBT> mCalibClocks{};  /// Calibration clock
  std::array<uint16_t, crateparams::sNELinksPerGBT> mLastClock{};    /// Last clock per link

  void addBoard(std::size_t ilink, const ELinkDecoder& decoder);
  void addLoc(std::size_t ilink, const ELinkDecoder& decoder);
  bool checkLoc(std::size_t ilink, const ELinkDecoder& decoder);
  bool updateIR(std::size_t ilink, const ELinkDecoder& decoder);
  bool invertPattern(LocalBoardRO& loc);
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_GBTOUTPUTHANDLER_H */
