// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_USERLOGIC_ENDPOINT_DECODER_H
#define O2_MCH_RAW_USERLOGIC_ENDPOINT_DECODER_H

#include <array>
#include "UserLogicElinkDecoder.h"
#include "MCHRawDecoder/DecodedDataHandlers.h"
#include <gsl/span>
#include <fmt/printf.h>
#include <fmt/format.h>
#include "MakeArray.h"
#include "Assertions.h"
#include <iostream>
#include <boost/multiprecision/cpp_int.hpp>
#include "MCHRawElecMap/FeeLinkId.h"
#include "PayloadDecoder.h"
#include "Debug.h"
#include "MCHRawCommon/DataFormats.h"

namespace o2::mch::raw
{

///
/// @brief A UserLogicEndpointDecoder groups 12 x (40 UserLogicElinkDecoder objects)
///

template <typename CHARGESUM, int VERSION>
class UserLogicEndpointDecoder : public PayloadDecoder<UserLogicEndpointDecoder<CHARGESUM, VERSION>>
{
 public:
  using ElinkDecoder = UserLogicElinkDecoder<CHARGESUM>;

  /// Constructor.
  /// \param linkId
  /// \param sampaChannelHandler the callable that will handle each SampaCluster
  UserLogicEndpointDecoder(uint16_t feeId,
                           std::function<std::optional<uint16_t>(FeeLinkId id)> fee2SolarMapper,
                           DecodedDataHandlers decodedDataHandlers);

  /** @name Main interface
    */
  ///@{

  /** @brief Append the equivalent n 64-bits words
    * bytes size (=n) must be a multiple of 8
    *
    * @return the number of bytes used in the bytes span
    */
  size_t append(Payload bytes);
  ///@}

  /** @name Methods for testing
    */

  ///@{

  /// Clear our internal Elinks
  void reset();
  ///@}

 private:
  uint16_t mFeeId;
  std::function<std::optional<uint16_t>(FeeLinkId id)> mFee2SolarMapper;
  DecodedDataHandlers mDecodedDataHandlers;
  std::map<uint16_t, std::array<ElinkDecoder, 40>> mElinkDecoders;
  int mNofGbtWordsSeens;
};

using namespace o2::mch::raw;
using namespace boost::multiprecision;

template <typename CHARGESUM, int VERSION>
UserLogicEndpointDecoder<CHARGESUM, VERSION>::UserLogicEndpointDecoder(uint16_t feeId,
                                                                       std::function<std::optional<uint16_t>(FeeLinkId id)> fee2SolarMapper,
                                                                       DecodedDataHandlers decodedDataHandlers)
  : PayloadDecoder<UserLogicEndpointDecoder<CHARGESUM, VERSION>>(decodedDataHandlers),
    mFeeId{feeId},
    mFee2SolarMapper{fee2SolarMapper},
    mDecodedDataHandlers(decodedDataHandlers),
    mNofGbtWordsSeens{0}
{
}

template <typename CHARGESUM, int VERSION>
size_t UserLogicEndpointDecoder<CHARGESUM, VERSION>::append(Payload buffer)
{
  if (buffer.size() % 8) {
    throw std::invalid_argument("buffer size should be a multiple of 8");
  }
  size_t n{0};

  for (size_t i = 0; i < buffer.size(); i += 8) {
    uint64_t word = (static_cast<uint64_t>(buffer[i + 0])) |
                    (static_cast<uint64_t>(buffer[i + 1]) << 8) |
                    (static_cast<uint64_t>(buffer[i + 2]) << 16) |
                    (static_cast<uint64_t>(buffer[i + 3]) << 24) |
                    (static_cast<uint64_t>(buffer[i + 4]) << 32) |
                    (static_cast<uint64_t>(buffer[i + 5]) << 40) |
                    (static_cast<uint64_t>(buffer[i + 6]) << 48) |
                    (static_cast<uint64_t>(buffer[i + 7]) << 56);

    if (word == 0) {
      continue;
    }
    if (word == 0xFEEDDEEDFEEDDEED) {
      continue;
    }

    ULHeaderWord<VERSION> ulword{word};

    int gbt = ulword.linkID;

    // The User Logic uses the condition gbt=15 to identify special control and diagnostics words
    // that are generated internally and do not contain data coming from the front-end electronics.
    if (gbt == 15) {
      // TODO: the exact format of the UL control words is still being defined and tested,
      // hence proper decoding will be implemented once the format is finalized.
      // For the moment we simply avoid throwing an exception when linkID is equal to 15
      continue;
    } else {
      if (gbt < 0 || gbt > 11) {
        SampaErrorHandler handler = mDecodedDataHandlers.sampaErrorHandler;
        if (handler) {
          DsElecId dsId{static_cast<uint16_t>(0), static_cast<uint8_t>(0), static_cast<uint8_t>(0)};
          handler(dsId, -1, ErrorBadLinkID);
        } else {
          throw fmt::format("warning : out-of-range gbt {} word={:08X}\n", gbt, word);
        }
      }
    }

    // Get the corresponding decoders array, or allocate it if does not exist yet
    auto d = mElinkDecoders.find(gbt);
    if (d == mElinkDecoders.end()) {

      // Compute the (feeId, linkId) pair...
      FeeLinkId feeLinkId(mFeeId, gbt);
      // ... and get the solar from it
      auto solarId = mFee2SolarMapper(feeLinkId);

      if (!solarId.has_value()) {
        SampaErrorHandler handler = mDecodedDataHandlers.sampaErrorHandler;
        if (handler) {
          DsElecId dsId{static_cast<uint16_t>(0), static_cast<uint8_t>(0), static_cast<uint8_t>(0)};
          handler(dsId, -1, ErrorUnknownLinkID);
        } else {
          throw std::logic_error(fmt::format("{} Could not get solarId from feeLinkId={}\n", __PRETTY_FUNCTION__, asString(feeLinkId)));
        }
      }

      mElinkDecoders.emplace(static_cast<uint16_t>(gbt),
                             impl::makeArray<40>([=](size_t i) {
                               DsElecId dselec{solarId.value(), static_cast<uint8_t>(i / 5), static_cast<uint8_t>(i % 5)};
                               return ElinkDecoder(dselec, mDecodedDataHandlers);
                             }));
      d = mElinkDecoders.find(gbt);
    }

    uint16_t dsid = ulword.dsID;
    if (dsid > 39) {
      SampaErrorHandler handler = mDecodedDataHandlers.sampaErrorHandler;
      if (handler) {
        DsElecId dsId{static_cast<uint16_t>(0), static_cast<uint8_t>(0), static_cast<uint8_t>(0)};
        handler(dsId, -1, ErrorBadELinkID);
      } else {
        throw fmt::format("warning : out-of-range DS ID {} word={:08X}\n", dsid, word);
      }
    }

    int8_t error = ulword.error;
    bool incomplete = ulword.incomplete > 0;
    uint64_t data50 = ulword.data;

    d->second.at(dsid).append(data50, error, incomplete);
    n += 8;
  }
  return n;
}

template <typename CHARGESUM, int VERSION>
void UserLogicEndpointDecoder<CHARGESUM, VERSION>::reset()
{
  for (auto& arrays : mElinkDecoders) {
    for (auto& d : arrays.second) {
      d.reset();
    }
  }
}
} // namespace o2::mch::raw
#endif
