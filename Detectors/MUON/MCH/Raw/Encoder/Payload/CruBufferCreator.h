// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHRawCommon/DataFormats.h"
#include "MCHRawEncoder/Encoder.h"
#include <vector>
#include <cstdint>

namespace o2::mch::raw::test
{

std::vector<uint8_t> fillChargeSum(Encoder& encoder, int norbit);

template <typename FORMAT, typename CHARGESUM>
struct CruBufferCreator {
  static std::vector<uint8_t> makeBuffer(int norbit = 1);
};

std::vector<uint8_t> fillChargeSum(Encoder& encoder);

template <typename FORMAT>
struct CruBufferCreator<FORMAT, ChargeSumMode> {
  static std::vector<uint8_t> makeBuffer(int norbit = 1)
  {
    auto encoder = createEncoder<FORMAT, ChargeSumMode, true>();

    return fillChargeSum(*(encoder.get()), norbit);
  }
};

} // namespace o2::mch::raw::test
