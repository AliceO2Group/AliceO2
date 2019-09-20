// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_ENCODER_BARE_ELINK_ENCODER_MERGER_H
#define O2_MCH_RAW_ENCODER_BARE_ELINK_ENCODER_MERGER_H

#include "ElinkEncoder.h"
#include "ElinkEncoderMerger.h"
#include "MCHRawCommon/DataFormats.h"
#include <gsl/span>

namespace o2::mch::raw
{

template <typename CHARGESUM>
bool areElinksAligned(gsl::span<ElinkEncoder<BareFormat, CHARGESUM>> elinks)
{
  auto len = elinks[0].len();
  for (auto i = 1; i < elinks.size(); i++) {
    if (elinks[i].len() != len) {
      return false;
    }
  }
  return true;
}

template <typename CHARGESUM>
void align(gsl::span<ElinkEncoder<BareFormat, CHARGESUM>> elinks)
{
  if (areElinksAligned(elinks)) {
    return;
  }
  auto e = std::max_element(elinks.begin(), elinks.end(),
                            [](const ElinkEncoder<BareFormat, CHARGESUM>& a, const ElinkEncoder<BareFormat, CHARGESUM>& b) {
                              return a.len() < b.len();
                            });

  // align all elink sizes by adding sync bits
  for (auto& elink : elinks) {
    elink.fillWithSync(e->len());
  }
}

template <typename CHARGESUM>
uint64_t aggregate(gsl::span<ElinkEncoder<BareFormat, CHARGESUM>> elinks, int jstart, int jend, int i)
{
  uint64_t w{0};
  for (int j = jstart; j < jend; j += 2) {
    for (int k = 0; k <= 1; k++) {
      bool v = elinks[j / 2].get(i + 1 - k);
      uint64_t mask = static_cast<uint64_t>(1) << (j + k);
      if (v) {
        w |= mask;
      } else {
        w &= ~mask;
      }
    }
  }
  return w;
}

template <typename CHARGESUM>
void elink2gbt(gsl::span<ElinkEncoder<BareFormat, CHARGESUM>> elinks, std::vector<uint64_t>& b64)
{
  int n = elinks[0].len();

  for (int i = 0; i < n - 1; i += 2) {
    uint64_t w0 = aggregate(elinks, 0, 64, i);
    uint64_t w1 = aggregate(elinks, 64, 80, i);
    b64.push_back(w0);
    b64.push_back(w1);
  }
}

template <typename CHARGESUM>
struct ElinkEncoderMerger<BareFormat, CHARGESUM> {
  void operator()(uint16_t gbtId,
                  gsl::span<ElinkEncoder<BareFormat, CHARGESUM>> elinks,
                  std::vector<uint64_t>& b64)
  {
    // align sizes of all elinks by adding sync bits
    align(elinks);

    // convert elinks to GBT words
    elink2gbt(elinks, b64);
  }
};

} // namespace o2::mch::raw
#endif
