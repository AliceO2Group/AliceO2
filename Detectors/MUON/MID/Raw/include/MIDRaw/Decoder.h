// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDRaw/Decoder.h
/// \brief  MID raw data decoder
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   18 November 2019
#ifndef O2_MID_DECODER_H
#define O2_MID_DECODER_H

#include <cstdint>
#include <vector>
#include <array>
#include <gsl/gsl>
#include "DataFormatsMID/ROFRecord.h"
#include "DetectorsRaw/RDHUtils.h"
#include "MIDRaw/CrateMasks.h"
#include "MIDRaw/CrateParameters.h"
#include "MIDRaw/ElectronicsDelay.h"
#include "MIDRaw/FEEIdConfig.h"
#include "MIDRaw/GBTBareDecoder.h"
#include "MIDRaw/GBTUserLogicDecoder.h"
#include "MIDRaw/LocalBoardRO.h"

namespace o2
{
namespace mid
{

template <typename GBTDECODER>
class Decoder
{
 public:
  Decoder();
  ~Decoder() = default;
  /// Sets the FEE ID config file
  void setFeeIdConfig(const FEEIdConfig& feeIdConfig) { mFEEIdConfig = feeIdConfig; }
  /// Sets the crate masks
  void setCrateMasks(const CrateMasks& masks) { mMasks = masks; }
  /// Sets the electronics delays
  void setElectronicsDelay(const ElectronicsDelay& electronicsDelay) { mElectronicsDelay = electronicsDelay; }
  void init(bool isDebugMode = false);
  void process(gsl::span<const uint8_t> bytes);
  template <typename RDH = o2::header::RAWDataHeader>
  void process(gsl::span<const uint8_t> payload, const RDH& rdh)
  {
    /// Processes the page
    uint16_t feeId = mFEEIdConfig.getFeeId(o2::raw::RDHUtils::getLinkID(rdh), o2::raw::RDHUtils::getEndPointID(rdh), o2::raw::RDHUtils::getCRUID(rdh));
    mGBTDecoders[feeId].process(payload, o2::raw::RDHUtils::getHeartBeatBC(rdh), o2::raw::RDHUtils::getHeartBeatOrbit(rdh), o2::raw::RDHUtils::getPageCounter(rdh));
  }
  /// Gets the vector of data
  const std::vector<LocalBoardRO>& getData() const { return mData; }

  /// Gets the vector of data RO frame records
  const std::vector<ROFRecord>& getROFRecords() const { return mROFRecords; }

  void flush();

  void clear();

  bool isComplete() const;

 private:
  std::vector<LocalBoardRO> mData{};                          /// Vector of output data
  std::vector<ROFRecord> mROFRecords{};                       /// List of ROF records
  std::array<GBTDECODER, crateparams::sNGBTs> mGBTDecoders{}; /// GBT decoders
  FEEIdConfig mFEEIdConfig{};                                 /// Crate FEEID mapper
  CrateMasks mMasks{};                                        /// Crate masks
  ElectronicsDelay mElectronicsDelay{};                       /// Delay in the electronics
};

} // namespace mid
} // namespace o2

#endif /* O2_MID_DECODER_H */
