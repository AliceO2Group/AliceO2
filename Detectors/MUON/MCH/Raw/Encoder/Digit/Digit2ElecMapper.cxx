// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHRawEncoderDigit/Digit2ElecMapper.h"

#include "MCHMappingInterface/Segmentation.h"
#include "Framework/Logger.h"
#include "MCHRawCommon/DataFormats.h"

namespace o2::mch::raw
{
// create a function that return the (DsElecId,dualSampaChannelId) of a digit
Digit2ElecMapper createDigit2ElecMapper(Det2ElecMapper det2elec)
{
  return [det2elec](const o2::mch::Digit& digit) -> std::optional<std::pair<DsElecId, DualSampaChannelId>> {
    auto deid = digit.getDetID();
    auto dsid = mapping::segmentation(deid).padDualSampaId(digit.getPadID());
    DsDetId detId{deid, dsid};
    auto dselocopt = det2elec(DsDetId(deid, dsid));
    if (!dselocopt.has_value()) {
      LOGP(warning, "got no location for (de,ds)=({},{})", deid, dsid);
      return std::nullopt;
    }
    DsElecId elecId = dselocopt.value();
    auto dualSampaChannelId = mapping::segmentation(deid).padDualSampaChannel(digit.getPadID());
    return std::make_pair(dselocopt.value(), dualSampaChannelId);
  };
}

} // namespace o2::mch::raw
