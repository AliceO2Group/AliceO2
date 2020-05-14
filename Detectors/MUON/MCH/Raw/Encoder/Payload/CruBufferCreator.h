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
#include "MCHRawEncoderPayload/PayloadEncoder.h"
#include <vector>
#include <cstdint>

namespace o2::mch::raw::test
{

template <typename FORMAT, typename CHARGESUM>
struct CruBufferCreator {
  static std::vector<std::byte> makeBuffer(int norbit = 1,
                                           uint32_t firstOrbit = 12345, uint16_t firstBC = 678);
};

} // namespace o2::mch::raw::test
