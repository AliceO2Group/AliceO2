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

/// \file   MID/Raw/src/LinkDecoder.cxx
/// \brief  Class interface for the MID GBT decoder
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   07 November 2020

#include "MIDRaw/LinkDecoder.h"

#include "DataFormatsMID/ROBoard.h"
#include "MIDRaw/CrateParameters.h"
#include "MIDRaw/ELinkManager.h"
#include "MIDRaw/FEEIdConfig.h"
#include "MIDRaw/Utils.h"

namespace o2
{
namespace mid
{
namespace impl
{

class GBTUserLogicDecoderImplV2
{
 public:
  GBTUserLogicDecoderImplV2(uint16_t ulFeeId, bool isDebugMode = false, const ElectronicsDelay& electronicsDelay = ElectronicsDelay(), const FEEIdConfig& feeIdConfig = FEEIdConfig())
  {
    mElinkManager.init(ulFeeId, isDebugMode, false, electronicsDelay, feeIdConfig);
    mELinkDecoder.setBareDecoder(false);
  }

  void operator()(gsl::span<const uint8_t> payload, uint32_t orbit, uint32_t trigger, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs)
  {
    /// Decodes the buffer
    mElinkManager.set(orbit, trigger);
    // for (auto& byte : payload) {
    auto it = payload.begin();
    auto end = payload.end();
    while (it != end) {
      if (mELinkDecoder.isZero(*it)) {
        // The e-link decoder is empty, meaning that we expect a new board.
        // The first byte of the board should have the STARTBIT on.
        // If this is not the case, we move to the next byte.
        // Notice that the payload has zeros until the end of the 256 bits word:
        // a) when moving from the regional high to regional low
        // b) at the end of the HBF
        ++it;
        continue;
      }

      if (mELinkDecoder.add(it, end)) {
        mElinkManager.onDone(mELinkDecoder, data, rofs);
        mELinkDecoder.reset();
      }
    }
  }

 private:
  ELinkDecoder mELinkDecoder{}; /// E-link decoder
  ELinkManager mElinkManager{}; /// ELinkManager
};

class GBTUserLogicDecoderImplV1
{
 public:
  GBTUserLogicDecoderImplV1(uint16_t feeId, bool isDebugMode = false, const ElectronicsDelay& electronicsDelay = ElectronicsDelay())
  {
    mCrateId = crateparams::getCrateIdFromGBTUniqueId(feeId);
    mOffset = 8 * crateparams::getGBTIdInCrate(feeId);
    mElinkManager.init(feeId, isDebugMode, true, electronicsDelay);
  }

  void operator()(gsl::span<const uint8_t> payload, uint32_t orbit, uint32_t trigger, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs)
  {
    /// Decodes the buffer
    mElinkManager.set(orbit, trigger);
    // for (auto& byte : payload) {
    auto it = payload.begin();
    auto end = payload.end();
    while (it != end) {
      if (mELinkDecoder.isZero(*it)) {
        // The e-link decoder is empty, meaning that we expect a new board.
        // The first byte of the board should have the STARTBIT on.
        // If this is not the case, we move to the next byte.
        // Notice that the payload has zeros until the end of the 256 bits word:
        // a) when moving from the regional high to regional low
        // b) at the end of the HBF
        ++it;
        continue;
      }

      if (mELinkDecoder.add(it, end)) {
        mElinkManager.onDone(mELinkDecoder, mCrateId, mOffset + mELinkDecoder.getId() % 8, data, rofs);
        mELinkDecoder.reset();
      }
    }
  }

 private:
  uint8_t mCrateId{0};          /// Crate ID
  uint8_t mOffset{0};           /// Loc ID offset
  ELinkDecoder mELinkDecoder{}; /// E-link decoder
  ELinkManager mElinkManager{}; /// ELinkManager
};

class GBTBareDecoderImplV1
{
 public:
  GBTBareDecoderImplV1(uint16_t feeId, bool isDebugMode = false, uint8_t mask = 0xFF, const ElectronicsDelay& electronicsDelay = ElectronicsDelay()) : mIsDebugMode(isDebugMode), mMask(mask)
  {
    /// Constructors
    mElinkManager.init(feeId, isDebugMode, true, electronicsDelay);
    auto crateId = crateparams::getCrateIdFromGBTUniqueId(feeId);
    auto offset = 8 * crateparams::getGBTIdInCrate(feeId);
    for (int ilink = 0; ilink < 10; ++ilink) {
      mBoardUniqueIds.emplace_back(raw::makeUniqueLocID(crateId, offset + ilink % 8));
    }
  }

  void operator()(gsl::span<const uint8_t> payload, uint32_t orbit, uint32_t trigger, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs)
  {
    /// Decodes the buffer
    mElinkManager.set(orbit, trigger);
    mData = &data;
    mROFs = &rofs;

    uint8_t byte = 0;
    size_t ilink = 0, linkMask = 0, byteOffset = 0;

    for (int ireg = 0; ireg < 2; ++ireg) {
      byteOffset = 5 * ireg;
      if (mIsDebugMode) {
        ilink = 8 + ireg;
        linkMask = (1 << ilink);
        for (size_t idx = byteOffset + 4; idx < payload.size(); idx += 16) {
          byte = payload[idx];
          if ((mIsFeeding & linkMask) || byte) {
            processRegDebug(ilink, byte);
          }
        }
      }
      for (int ib = 0; ib < 4; ++ib) {
        ilink = ib + 4 * ireg;
        linkMask = (1 << ilink);
        if (mMask & linkMask) {
          for (size_t idx = byteOffset + ib; idx < payload.size(); idx += 16) {
            byte = payload[idx];
            if ((mIsFeeding & linkMask) || byte) {
              processLoc(ilink, byte);
            }
          }
        }
      }
    }
  }

 private:
  void processLoc(size_t ilink, uint8_t byte)
  {
    /// Processes the local board information
    auto& decoder = mElinkManager.getDecoder(mBoardUniqueIds[ilink], true);
    if (decoder.getNBytes() > 0) {
      decoder.addAndComputeSize(byte);
      if (decoder.isComplete()) {
        mElinkManager.onDone(decoder, mBoardUniqueIds[ilink], *mData, *mROFs);
        decoder.reset();
        mIsFeeding &= (~(1 << ilink));
      }
    } else if ((byte & (raw::sSTARTBIT | raw::sCARDTYPE)) == (raw::sSTARTBIT | raw::sCARDTYPE)) {
      decoder.add(byte);
      mIsFeeding |= (1 << ilink);
    }
  }

  void processRegDebug(size_t ilink, uint8_t byte)
  {
    /// Processes the regional board information in debug mode
    auto& decoder = mElinkManager.getDecoder(mBoardUniqueIds[ilink], false);
    if (decoder.getNBytes() > 0) {
      decoder.add(byte);
      if (decoder.isComplete()) {
        mElinkManager.onDone(decoder, mBoardUniqueIds[ilink], *mData, *mROFs);
        decoder.reset();
        mIsFeeding &= (~(1 << ilink));
      }
    } else if (byte & raw::sSTARTBIT) {
      decoder.add(byte);
      mIsFeeding |= (1 << ilink);
    }
  }

  bool mIsDebugMode{false};               /// Debug mode
  uint8_t mMask{0xFF};                    /// GBT mask
  uint16_t mIsFeeding{0};                 /// Flag to check if the e-link is feeding
  ELinkManager mElinkManager{};           /// ELinkManager
  std::vector<uint8_t> mBoardUniqueIds{}; /// Unique board IDs
  std::vector<ROBoard>* mData{nullptr};   ///! Data not owner
  std::vector<ROFRecord>* mROFs{nullptr}; ///! ROFRecords not owner
};

} // namespace impl

void LinkDecoder::process(gsl::span<const uint8_t> payload, uint32_t orbit, uint32_t trigger, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs)
{
  /// Decodes the data
  mDecode(payload, orbit, trigger, data, rofs);
}

std::unique_ptr<LinkDecoder> createGBTDecoder(const o2::header::RDHAny& rdh, uint16_t feeId, bool isDebugMode, uint8_t mask, const ElectronicsDelay& electronicsDelay)
{
  /// Creates the correct implementation of the GBT decoder
  if (raw::isBare(rdh)) {
    return std::make_unique<LinkDecoder>(impl::GBTBareDecoderImplV1(feeId, isDebugMode, mask, electronicsDelay));
  }
  std::cout << "Error: GBT decoder not defined for UL. Please use Link decoder instead" << std::endl;
  return nullptr;
}

std::unique_ptr<LinkDecoder> createLinkDecoder(const o2::header::RDHAny& rdh, uint16_t feeId, bool isDebugMode, uint8_t mask, const ElectronicsDelay& electronicsDelay, const FEEIdConfig& feeIdConfig)
{
  /// Creates the correct implementation of the GBT decoder
  if (raw::isBare(rdh)) {
    return std::make_unique<LinkDecoder>(impl::GBTBareDecoderImplV1(feeId, isDebugMode, mask, electronicsDelay));
  }
  return std::make_unique<LinkDecoder>(impl::GBTUserLogicDecoderImplV2(feeId, isDebugMode, electronicsDelay, feeIdConfig));
}

std::unique_ptr<LinkDecoder> createLinkDecoder(uint16_t feeId, bool isBare, bool isDebugMode, uint8_t mask, const ElectronicsDelay& electronicsDelay, const FEEIdConfig& feeIdConfig)
{
  /// Creates the correct implementation of the GBT decoder
  if (isBare) {
    return std::make_unique<LinkDecoder>(impl::GBTBareDecoderImplV1(feeId, isDebugMode, mask, electronicsDelay));
  }
  return std::make_unique<LinkDecoder>(impl::GBTUserLogicDecoderImplV2(feeId, isDebugMode, electronicsDelay, feeIdConfig));
}

} // namespace mid
} // namespace o2
