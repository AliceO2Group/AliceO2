// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_ELECMAP_DS_DET_ID_H
#define O2_MCH_RAW_ELECMAP_DS_DET_ID_H

#include <cstdint>
#include <iosfwd>

namespace o2::mch::raw
{
/// A DsDetId is just a pair (detection element id, dual sampa id)
class DsDetId
{
 public:
  DsDetId(int deId, int dsId);

  /// deId returns one of the 156 possible detection element id
  uint16_t deId() const { return mDeId; }

  /// dsId returns a dual sampa id
  /// note that dsId >= 1024 means a dual sampa on the non-bending plane
  uint16_t dsId() const { return mDsId; }

 private:
  uint16_t mDeId;
  uint16_t mDsId;
};

/// Create a DsDetId object from a integer code
DsDetId decodeDsDetId(uint32_t code);

/// Create an integer code for the given id
uint32_t encode(const DsDetId& id);

/// Returns a string representation of the id
std::string asString(DsDetId dsDetId);

std::ostream& operator<<(std::ostream& os, const DsDetId& id);

} // namespace o2::mch::raw
#endif
