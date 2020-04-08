// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_ELECMAP_FEE_LINK_ID_H
#define O2_MCH_RAW_ELECMAP_FEE_LINK_ID_H

#include <cstdint>
#include <iosfwd>

namespace o2::mch::raw
{

class FeeLinkId
{
 public:
  FeeLinkId(uint16_t feeId, uint8_t linkId);

  uint16_t feeId() const { return mFeeId; }
  uint8_t linkId() const { return mLinkId; }

 private:
  uint16_t mFeeId;
  uint8_t mLinkId;
};

FeeLinkId decodeFeeLinkId(uint32_t code);

uint32_t encode(const FeeLinkId& id);

bool operator<(const FeeLinkId& f1, const FeeLinkId& f2);

std::ostream& operator<<(std::ostream& os, const FeeLinkId& id);

std::string asString(const FeeLinkId& feeLinkId);
} // namespace o2::mch::raw

#endif
