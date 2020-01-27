// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_USERLOGIC_GBT_DECODER_H
#define O2_MCH_RAW_USERLOGIC_GBT_DECODER_H

#include <array>
#include "UserLogicElinkDecoder.h"
#include "MCHRawDecoder/SampaChannelHandler.h"
#include <gsl/span>
#include "UserLogicGBTDecoder.h"
#include <fmt/printf.h>
#include <fmt/format.h>
#include "MakeArray.h"
#include "Assertions.h"
#include <iostream>
#include <boost/multiprecision/cpp_int.hpp>

namespace o2::mch::raw
{

/// @brief A UserLogicGBTDecoder groups 40 UserLogicChannelDecoder objects.

template <typename CHARGESUM>
class UserLogicGBTDecoder
{
 public:
  static constexpr uint8_t baseSize{64};
  /// Constructor.
  /// \param solarId
  /// \param sampaChannelHandler the callable that will handle each SampaCluster
  UserLogicGBTDecoder(uint16_t solarId, SampaChannelHandler sampaChannelHandler);

  /** @name Main interface 
    */
  ///@{

  /** @brief Append the equivalent n 64-bits words 
    * bytes size (=n) must be a multiple of 8
    *
    * @return the number of bytes used in the bytes span
    */
  size_t append(gsl::span<uint8_t> bytes);
  ///@}

  /** @name Methods for testing
    */

  ///@{

  /// Clear our internal Elinks
  void reset() {}
  ///@}

 private:
  int mSolarId;
  std::array<UserLogicElinkDecoder<CHARGESUM>, 40> mElinkDecoders;
  int mNofGbtWordsSeens;
};

using namespace o2::mch::raw;
using namespace boost::multiprecision;

template <typename CHARGESUM>
UserLogicGBTDecoder<CHARGESUM>::UserLogicGBTDecoder(uint16_t solarId,
                                                    SampaChannelHandler sampaChannelHandler)
  : mSolarId{solarId},
    mElinkDecoders{
      impl::makeArray<40>([=](size_t i) {
        return UserLogicElinkDecoder<CHARGESUM>(DsElecId{solarId, static_cast<uint8_t>(i / 8), static_cast<uint8_t>(i % 5)}, sampaChannelHandler);
      })} // namespace o2::mch::raw
    ,
    mNofGbtWordsSeens{0}
{
}

template <typename CHARGESUM>
size_t UserLogicGBTDecoder<CHARGESUM>::append(gsl::span<uint8_t> buffer)
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
    int gbt = (word >> 59) & 0x1F;
    if (gbt != mSolarId) {
      std::cout << fmt::format("warning : solarId {} != expected {} word={:08X}\n", gbt, mSolarId, word);
      // throw std::invalid_argument(fmt::format("gbt {} != expected {} word={:X}\n", gbt, mGbtId, word));
    }

    uint16_t dsid = (word >> 53) & 0x3F;
    impl::assertIsInRange("dsid", dsid, 0, 39);

    // the remaining 50 bits are passed to the ElinkDecoder
    uint64_t data = word & UINT64_C(0x003FFFFFFFFFFFFF);
    mElinkDecoders.at(dsid).append(data);
    n += 8;
  }
  return n;
}

} // namespace o2::mch::raw
#endif
