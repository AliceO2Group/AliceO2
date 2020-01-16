// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_ELECMAP_CRU_LINK_ID_H
#define O2_MCH_RAW_ELECMAP_CRU_LINK_ID_H

#include <cstdint>
#include <iosfwd>

namespace o2::mch::raw
{
class CruLinkId
{
 public:
  CruLinkId(uint16_t cruId, uint8_t linkId);
  CruLinkId(uint16_t cruId, uint8_t linkId, uint16_t deId);

  uint16_t cruId() const { return mCruId; }
  uint16_t deId() const { return mDeId; }
  uint8_t linkId() const { return mLinkId; }

 private:
  uint16_t mCruId;
  uint8_t mLinkId;
  uint16_t mDeId;
};

CruLinkId decodeCruLinkId(uint32_t code);
uint32_t encode(const CruLinkId& id);

std::ostream& operator<<(std::ostream& os, const CruLinkId& id);

} // namespace o2::mch::raw
#endif
