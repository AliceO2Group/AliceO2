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

#include "MCHROFFiltering/IRFrameFilter.h"
#include "CommonDataFormat/InteractionRecord.h"

using o2::InteractionRecord;
using o2::dataformats::IRFrame;

namespace o2::mch
{
ROFFilter createIRFrameFilter(gsl::span<const o2::dataformats::IRFrame> irframes)
{
  return [irframes](const ROFRecord& rof) {
    InteractionRecord rofStart{rof.getBCData()};
    InteractionRecord rofEnd = rofStart + rof.getBCWidth() - 1;
    IRFrame ref(rofStart, rofEnd);
    for (const auto& ir : irframes) {
      if (ir.info > 0) {
        auto overlap = ref.getOverlap(ir);
        if (overlap.isValid()) {
          return true;
        }
      }
    }
    return false;
  };
}
} // namespace o2::mch
