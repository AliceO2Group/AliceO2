// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_BARE_GBT_DECODER_H
#define O2_MCH_RAW_BARE_GBT_DECODER_H

#include <array>
#include "BareElinkDecoder.h"
#include "MCHRawDecoder/SampaChannelHandler.h"
#include <gsl/span>
#include <fmt/printf.h>
#include <fmt/format.h>
#include "MakeArray.h"
#include "Assertions.h"
#include <iostream>
#include <boost/multiprecision/cpp_int.hpp>
#include "PayloadDecoder.h"

namespace o2
{
namespace mch
{
namespace raw
{

/// @brief A BareGBTDecoder groups 40 ElinkDecoder objects.
///
/// It's one GBT = one Solar.

template <typename CHARGESUM>
class BareGBTDecoder : public PayloadDecoder<BareGBTDecoder<CHARGESUM>>
{
 public:
  /// Constructor.
  /// \param solarId
  /// \param sampaChannelHandler the callable that will handle each SampaCluster
  BareGBTDecoder(uint16_t solarId, SampaChannelHandler sampaChannelHandler);

  /** @name Main interface 
    */
  ///@{

  /** @brief Append the equivalent n GBT words 
    * (n x 128 bits, split in 16 bytes).
    * bytes size (=n) must be a multiple of 16
    * Given that the MCH data only uses 80 out of the 128 bits
    * only the 10 first bytes of each group of 16 are used
    * 
    * @return the number of bytes that have been used from bytes span
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
  void append(uint32_t, int, int);

 private:
  int mSolarId;
  std::array<BareElinkDecoder<CHARGESUM>, 40> mElinks;
  int mNofGbtWordsSeens;
  uint32_t mOrbit;
};

using namespace boost::multiprecision;

template <typename CHARGESUM>
BareGBTDecoder<CHARGESUM>::BareGBTDecoder(uint16_t solarId,
                                          SampaChannelHandler sampaChannelHandler)
  : PayloadDecoder<BareGBTDecoder<CHARGESUM>>(sampaChannelHandler),
    mSolarId{solarId},
    mElinks{impl::makeArray<40>([=](uint8_t i) { return BareElinkDecoder<CHARGESUM>(DsElecId{solarId, static_cast<uint8_t>(i / 5), static_cast<uint8_t>(i % 5)}, sampaChannelHandler); })},
    mNofGbtWordsSeens{0}
{
}

template <typename CHARGESUM>
size_t BareGBTDecoder<CHARGESUM>::append(Payload bytes)
{
  if (bytes.size() % 16 != 0) {
    throw std::invalid_argument("can only bytes by group of 16 (i.e. 128 bits)");
  }
  size_t n{0};
  for (int j = 0; j < bytes.size(); j += 16) {
    ++mNofGbtWordsSeens;
    int elinkIndex = 0;
    for (auto b : bytes.subspan(j, 10)) {
      const uint8_t b8 = std::to_integer<uint8_t>(b);
      mElinks[elinkIndex++].append(b8 & 2, b8 & 1);
      mElinks[elinkIndex++].append(b8 & 8, b8 & 4);
      mElinks[elinkIndex++].append(b8 & 32, b8 & 16);
      mElinks[elinkIndex++].append(b8 & 128, b8 & 64);
    }
    n += 10;
  }
  return n;
}

template <typename CHARGESUM>
void BareGBTDecoder<CHARGESUM>::reset()
{
  for (auto& e : mElinks) {
    e.reset();
  }
}
} // namespace raw
} // namespace mch
} // namespace o2

#endif
