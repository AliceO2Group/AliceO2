// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDRaw/CRUBareDecoder.h
/// \brief  MID CRU core decoder
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   18 November 2019
#ifndef O2_MID_CRUBAREDECODER_H
#define O2_MID_CRUBAREDECODER_H

#include <cstdint>
#include <vector>
#include <gsl/gsl>
#include "DataFormatsMID/ROFRecord.h"
#include "MIDRaw/CrateMapper.h"
#include "MIDRaw/CrateParameters.h"
#include "MIDRaw/ELinkDecoder.h"
#include "MIDRaw/LocalBoardRO.h"
#include "MIDRaw/RawBuffer.h"

namespace o2
{
namespace mid
{
class CRUBareDecoder
{
 public:
  void init(bool debugMode = false);
  void process(gsl::span<const uint8_t> bytes);
  /// Gets the vector of data
  const std::vector<LocalBoardRO>& getData() const { return mData; }

  /// Gets the vector of data RO frame records
  const std::vector<ROFRecord>& getROFRecords() const { return mROFRecords; }

  bool isComplete() const;

 private:
  RawBuffer<uint8_t> mBuffer{};                                           /// Raw buffer handler
  std::vector<LocalBoardRO> mData{};                                      /// Vector of output data
  std::vector<ROFRecord> mROFRecords{};                                   /// List of ROF records
  CrateMapper mCrateMapper{};                                             /// Crate mapper
  uint8_t mCrateId{0};                                                    /// Crate ID
  std::array<uint16_t, crateparams::sNELinksPerGBT> mCalibClocks{};       /// Calibration clock
  std::array<ELinkDecoder, crateparams::sNELinksPerGBT> mELinkDecoders{}; /// E-link decoders
  std::function<void(size_t)> mAddReg{[](size_t) {}};                     ///! Add regional board

  std::function<bool(size_t)> mCheckBoard{std::bind(&CRUBareDecoder::checkBoard, this, std::placeholders::_1)}; ///! Check board

  bool nextGBTWord();
  void processGBT(size_t offset);
  void reset();
  void addBoard(size_t ilink);
  bool checkBoard(size_t ilink);
  void addLoc(size_t ilink);
  uint16_t getPattern(uint16_t pattern, bool invert) const;
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_CRUBAREDECODER_H */
