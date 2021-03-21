// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Raw/src/GBTDecoder.cxx
/// \brief  Class interface for the MID GBT decoder
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   07 November 2020

#include "MIDRaw/GBTDecoder.h"

#include "MIDRaw/FEEIdConfig.h"
#include "MIDRaw/GBTOutputHandler.h"
#include "DataFormatsMID/ROBoard.h"

namespace o2
{
namespace mid
{
namespace impl
{
class GBTUserLogicDecoderImpl
{
 public:
  GBTUserLogicDecoderImpl(uint16_t feeId, bool isDebugMode = false, uint8_t mask = 0xFF, const ElectronicsDelay& electronicsDelay = ElectronicsDelay())
  {
    mOutputHandler.setFeeId(feeId);
    mOutputHandler.setElectronicsDelay(electronicsDelay);
    if (isDebugMode) {
      mOnDoneLoc = &GBTOutputHandler::onDoneLocDebug;
      mOnDoneReg = &GBTOutputHandler::onDoneRegDebug;
    }
  }

  void operator()(gsl::span<const uint8_t> payload, uint32_t orbit, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs)
  {
    /// Decodes the buffer
    mOutputHandler.set(orbit, data, rofs);
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
        if (raw::isLoc(mELinkDecoder.getStatusWord())) {
          std::invoke(mOnDoneLoc, mOutputHandler, mELinkDecoder.getId() % 8, mELinkDecoder);
        } else {
          size_t ilink = 8 + mELinkDecoder.getId() % 8;
          if (ilink <= 9) {
            std::invoke(mOnDoneReg, mOutputHandler, ilink, mELinkDecoder);
          }
        }
        mELinkDecoder.reset();
      }
    }
  }

 private:
  GBTOutputHandler mOutputHandler{}; /// GBT output handler
  ELinkDecoder mELinkDecoder{};      /// E-link decoder

  typedef void (GBTOutputHandler::*OnDoneFunction)(size_t, const ELinkDecoder&);

  OnDoneFunction mOnDoneLoc{&GBTOutputHandler::onDoneLoc}; ///! Processes the local board
  OnDoneFunction mOnDoneReg{&GBTOutputHandler::onDoneReg}; ///! Processes the regional board
};

class GBTBareDecoderImpl
{
 public:
  GBTBareDecoderImpl(uint16_t feeId, bool isDebugMode = false, uint8_t mask = 0xFF, const ElectronicsDelay& electronicsDelay = ElectronicsDelay()) : mIsDebugMode(isDebugMode), mMask(mask)
  {
    /// Constructor
    mOutputHandler.setFeeId(feeId);
    mOutputHandler.setElectronicsDelay(electronicsDelay);
    if (isDebugMode) {
      mOnDoneLoc = &GBTOutputHandler::onDoneLocDebug;
    }
  }

  void operator()(gsl::span<const uint8_t> payload, uint32_t orbit, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs)
  {
    /// Decodes the buffer
    mOutputHandler.set(orbit, data, rofs);

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
    if (mELinkDecoders[ilink].getNBytes() > 0) {
      mELinkDecoders[ilink].addAndComputeSize(byte);
      if (mELinkDecoders[ilink].isComplete()) {
        std::invoke(mOnDoneLoc, mOutputHandler, ilink, mELinkDecoders[ilink]);
        mELinkDecoders[ilink].reset();
        mIsFeeding &= (~(1 << ilink));
      }
    } else if ((byte & (raw::sSTARTBIT | raw::sCARDTYPE)) == (raw::sSTARTBIT | raw::sCARDTYPE)) {
      mELinkDecoders[ilink].add(byte);
      mIsFeeding |= (1 << ilink);
    }
  }

  void processRegDebug(size_t ilink, uint8_t byte)
  {
    /// Processes the regional board information in debug mode
    if (mELinkDecoders[ilink].getNBytes() > 0) {
      mELinkDecoders[ilink].add(byte);
      if (mELinkDecoders[ilink].isComplete()) {
        mOutputHandler.onDoneRegDebug(ilink, mELinkDecoders[ilink]);
        mELinkDecoders[ilink].reset();
        mIsFeeding &= (~(1 << ilink));
      }
    } else if (byte & raw::sSTARTBIT) {
      mELinkDecoders[ilink].add(byte);
      mIsFeeding |= (1 << ilink);
    }
  }

  bool mIsDebugMode{false};          /// Debug mode
  uint8_t mMask{0xFF};               /// GBT mask
  uint16_t mIsFeeding{0};            /// Flag to check if the e-link is feeding
  GBTOutputHandler mOutputHandler{}; /// GBT output handler

  std::array<ELinkDecoder, crateparams::sNELinksPerGBT> mELinkDecoders{}; /// E-link decoders

  typedef void (GBTOutputHandler::*OnDoneFunction)(size_t, const ELinkDecoder&);

  OnDoneFunction mOnDoneLoc{&GBTOutputHandler::onDoneLoc}; ///! Processes the local board
};

/// Alternative bare decoder implementation
/// Data are first ranged per link and then each link is decoded in a similar way to what is done for the user logic
/// CAVEAT: abandoned since filling the vector per link is much slower
/// Kept here for future reference (parallelization?)
class GBTBareDecoderLinkImpl
{
 public:
  GBTBareDecoderLinkImpl(uint16_t feeId, bool isDebugMode = false, uint8_t mask = 0xFF, const ElectronicsDelay& electronicsDelay = ElectronicsDelay()) : mIsDebugMode(isDebugMode), mMask(mask)
  {
    /// Constructor
    mOutputHandler.setFeeId(feeId);
    mOutputHandler.setElectronicsDelay(electronicsDelay);
    mIsDebugMode = isDebugMode;
    if (isDebugMode) {
      mOnDoneLoc = &GBTOutputHandler::onDoneLocDebug;
    }
    mLinkPayload.reserve(0x20000);
  }

  void operator()(gsl::span<const uint8_t> payload, uint32_t orbit, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs)
  {
    /// Decodes the buffer
    mOutputHandler.set(orbit, data, rofs);

    for (int ireg = 0; ireg < 2; ++ireg) {
      size_t byteOffset = 5 * ireg;

      if (mIsDebugMode) {
        // Treat regional cards
        size_t ilink = 8 + ireg;
        mLinkPayload.clear();
        for (size_t idx = byteOffset + 4, end = payload.size(); idx < end; idx += 16) {
          mLinkPayload.emplace_back(payload[idx]);
        }
        auto it = mLinkPayload.begin();
        auto end = mLinkPayload.end();
        while (it != end) {
          if (mELinkDecoders[ilink].isZero(*it)) {
            ++it;
            continue;
          }
          if (mELinkDecoders[ilink].addCore(it, end)) {
            mOutputHandler.onDoneRegDebug(ilink, mELinkDecoders[ilink]);
            mELinkDecoders[ilink].reset();
          }
        }
      }

      // Treat local cards
      for (int ib = 0; ib < 4; ++ib) {
        size_t ilink = ib + 4 * ireg;
        if (mMask & (1 << ilink)) {
          mLinkPayload.clear();
          for (size_t idx = byteOffset + ib, end = payload.size(); idx < end; idx += 16) {
            mLinkPayload.emplace_back(payload[idx]);
          }
          auto it = mLinkPayload.begin();
          auto end = mLinkPayload.end();
          while (it != end) {
            if (mELinkDecoders[ilink].isZero(*it)) {
              ++it;
              continue;
            }
            if (mELinkDecoders[ilink].add(it, end)) {
              std::invoke(mOnDoneLoc, mOutputHandler, mELinkDecoders[ilink].getId() % 8, mELinkDecoders[ilink]);
              mELinkDecoders[ilink].reset();
            }
          }
        }
      }
    }
  }

 private:
  bool mIsDebugMode{false};          /// Debug mode
  uint8_t mMask{0xFF};               /// GBT mask
  GBTOutputHandler mOutputHandler{}; /// GBT output handler

  std::array<ELinkDecoder, crateparams::sNELinksPerGBT> mELinkDecoders{}; /// E-link decoders
  std::vector<uint8_t> mLinkPayload{};                                    /// Link payload

  typedef void (GBTOutputHandler::*OnDoneFunction)(size_t, const ELinkDecoder&);

  OnDoneFunction mOnDoneLoc{&GBTOutputHandler::onDoneLoc}; ///! Processes the local board
};

/// Alternative implementation of the bare decoder
/// When a start bit is found, we try to add all of the expected bytes.
/// This should in principle allow to avoid performing a check on the expected data size at each newly added byte.
/// But tests show that the implementation is slightly slower than the standard one.
class GBTBareDecoderInsertImpl
{
 public:
  GBTBareDecoderInsertImpl(uint16_t feeId, bool isDebugMode = false, uint8_t mask = 0xFF, const ElectronicsDelay& electronicsDelay = ElectronicsDelay()) : mIsDebugMode(isDebugMode), mMask(mask)
  {
    /// Constructor
    mOutputHandler.setFeeId(feeId);
    mOutputHandler.setElectronicsDelay(electronicsDelay);
    if (isDebugMode) {
      mOnDoneLoc = &GBTOutputHandler::onDoneLocDebug;
    }
  }

  void operator()(gsl::span<const uint8_t> payload, uint32_t orbit, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs)
  {
    /// Decodes the buffer
    mOutputHandler.set(orbit, data, rofs);

    size_t step = 16;
    size_t end = payload.size();

    for (int ireg = 0; ireg < 2; ++ireg) {
      size_t byteOffset = 5 * ireg;

      if (mIsDebugMode) {
        // Treat regional cards
        size_t ilink = 8 + ireg;
        size_t idx = byteOffset + 4;

        while (idx < end) {
          if (mELinkDecoders[ilink].isZero(payload[idx])) {
            idx += step;
          } else if (mELinkDecoders[ilink].addCore(idx, payload, step)) {
            mOutputHandler.onDoneRegDebug(ilink, mELinkDecoders[ilink]);
            mELinkDecoders[ilink].reset();
          }
        }
      }

      // Treat local cards
      for (int ib = 0; ib < 4; ++ib) {
        size_t ilink = ib + 4 * ireg;
        if (mMask & (1 << ilink)) {
          size_t idx = byteOffset + ib;
          while (idx < end) {
            if (mELinkDecoders[ilink].isZero(payload[idx])) {
              idx += step;
            } else if (mELinkDecoders[ilink].add(idx, payload, step)) {
              std::invoke(mOnDoneLoc, mOutputHandler, mELinkDecoders[ilink].getId() % 8, mELinkDecoders[ilink]);
              mELinkDecoders[ilink].reset();
            }
          }
        }
      }
    }
  }

 private:
  uint8_t mMask{0xFF};               /// GBT mask
  bool mIsDebugMode{false};          /// Debug mode
  GBTOutputHandler mOutputHandler{}; /// GBT output handler

  std::array<ELinkDecoder, crateparams::sNELinksPerGBT> mELinkDecoders{}; /// E-link decoders

  typedef void (GBTOutputHandler::*OnDoneFunction)(size_t, const ELinkDecoder&);

  OnDoneFunction mOnDoneLoc{&GBTOutputHandler::onDoneLoc}; ///! Processes the local board
};

} // namespace impl

void GBTDecoder::process(gsl::span<const uint8_t> payload, uint32_t orbit, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs)
{
  /// Decodes the data
  mDecode(payload, orbit, data, rofs);
}

std::unique_ptr<GBTDecoder> createGBTDecoder(const o2::header::RDHAny& rdh, uint16_t feeId, bool isDebugMode, uint8_t mask, const ElectronicsDelay& electronicsDelay)
{
  /// Creates the correct implementation of the GBT decoder
  if (o2::raw::RDHUtils::getLinkID(rdh) == raw::sUserLogicLinkID) {
    return std::make_unique<GBTDecoder>(impl::GBTUserLogicDecoderImpl(feeId, isDebugMode, mask, electronicsDelay));
  }
  return std::make_unique<GBTDecoder>(impl::GBTBareDecoderImpl(feeId, isDebugMode, mask, electronicsDelay));
}

std::unique_ptr<GBTDecoder> createGBTDecoder(uint16_t feeId, bool isBare, bool isDebugMode, uint8_t mask, const ElectronicsDelay& electronicsDelay)
{
  /// Creates the correct implementation of the GBT decoder
  if (isBare) {
    return std::make_unique<GBTDecoder>(impl::GBTBareDecoderImpl(feeId, isDebugMode, mask, electronicsDelay));
  }
  return std::make_unique<GBTDecoder>(impl::GBTUserLogicDecoderImpl(feeId, isDebugMode, mask, electronicsDelay));
}

} // namespace mid
} // namespace o2
