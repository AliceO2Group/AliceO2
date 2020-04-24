// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_WORKFLOW_MAPCRU_H
#define O2_MCH_WORKFLOW_MAPCRU_H

#include <string_view>
#include <array>
#include <cstdlib>
#include <optional>
#include "MCHRawElecMap/FeeLinkId.h"

namespace o2::mch::raw
{
class MapCRU
{
 public:
  MapCRU(std::string_view content);
  std::optional<uint16_t> operator()(const FeeLinkId& feeLinkId) const;
  size_t size() const;

 private:
  int indexFeeLink(int feeid, int linkid) const;

 private:
  static constexpr int sMaxFeeId = 64;
  static constexpr int sMaxLinkId = 12;
  std::array<uint16_t, sMaxFeeId * sMaxLinkId> mFeeLink2Solar;
};

} // namespace o2::mch::raw
#endif
