// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_ELINK_ENCODER_H
#define O2_MCH_RAW_ELINK_ENCODER_H

#include <vector>
#include <cstdint>
#include <gsl/span>

namespace o2::mch::raw
{
class SampaCluster;
class SampaHeader;

template <typename FORMAT, typename CHARGESUM>
class ElinkEncoder
{
 public:
  explicit ElinkEncoder(uint8_t elinkId, uint8_t chip, int phase = 0);

  void addChannelData(uint8_t chId, const std::vector<SampaCluster>& data);

  size_t moveToBuffer(std::vector<uint64_t>& buffer, uint64_t prefix);

  void clear();
};

SampaHeader buildHeader(uint8_t elinkId, uint8_t chId, const std::vector<SampaCluster>& data);

} // namespace o2::mch::raw

#endif
