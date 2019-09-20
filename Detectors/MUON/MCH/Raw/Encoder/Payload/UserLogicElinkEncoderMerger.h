// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_ENCODER_USER_LOGIC_ENCODER_MERGER_H
#define O2_MCH_RAW_ENCODER_USER_LOGIC_ENCODER_MERGER_H

#include "UserLogicElinkEncoder.h"
#include "MCHRawCommon/DataFormats.h"
#include <fmt/format.h>

namespace o2::mch::raw
{

template <typename CHARGESUM>
struct ElinkEncoderMerger<UserLogicFormat, CHARGESUM> {

  void operator()(uint16_t gbtId,
                  gsl::span<ElinkEncoder<UserLogicFormat, CHARGESUM>> elinks,
                  std::vector<uint64_t>& b64)
  {
    const uint64_t gbtIdMask((static_cast<uint64_t>(gbtId & 0x1F) << 59));
    for (auto& elink : elinks) {
      elink.moveToBuffer(b64, gbtIdMask);
    }
  }
};
} // namespace o2::mch::raw
#endif
