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

#ifndef O2_MCH_CONDITIONS_STATUSMAP_H
#define O2_MCH_CONDITIONS_STATUSMAP_H

#include "MCHGlobalMapping/ChannelCode.h"
#include "DataFormatsMCH/DsChannelId.h"
#include <cstdint>
#include <gsl/span>
#include <map>
#include <utility>
#include <vector>

namespace o2::mch
{

/** The StatusMap contains the list of MCH channels that are not nominal.
 * Said otherwise, it's a list of potential bad channels.
 *
 * Each potentially bad channel is ascribed a 32-bits mask that indicate
 * the source of information used to incriminate it.
 *
 * So far only two sources exist :
 * - kBadPedestal : the list generated at each pedestal run at Pt2
 * - kRejectList : a (manual) list
 *
 * In the future (based on our experience during Run1,2), we'll most probably
 * need to add information from the DCS HV (and possibly LV) values as well.
 *
 */
class StatusMap
{
 public:
  enum Status : uint32_t {
    kOK = 0,
    kBadPedestal = 1 << 0,
    kRejectList = 1 << 1
  };

  using iterator = std::map<ChannelCode, uint32_t>::iterator;
  using const_iterator = std::map<ChannelCode, uint32_t>::const_iterator;

  iterator begin() { return mStatus.begin(); }
  iterator end() { return mStatus.end(); }
  const_iterator begin() const { return mStatus.begin(); }
  const_iterator end() const { return mStatus.end(); }

  /** add all the badchannels referenced using DsChannelId (aka readout
   * channel id, aka (solar,group,index)) to this status map,
   * assigning them the corresponding mask.
   * @throw runtime_error if the mask is invalid
   */
  void add(gsl::span<const DsChannelId> badchannels, uint32_t mask);

  /** add all the badchannels referenced using ChannelCode
   * to this status map, assigning them the corresponding mask.
   * @throw runtime_error if the mask is invalid
   */
  void add(gsl::span<const ChannelCode> badchannels, uint32_t mask);

  /** whether or not this statusmap contains no (potentially) bad channels */
  bool empty() const { return mStatus.empty(); }

  /** clear the content of the statusmap (after clear the statusmap is thus empty) */
  void clear() { mStatus.clear(); }

  /** return the status of a given channel */
  uint32_t status(const ChannelCode& id) const;

 private:
  std::map<ChannelCode, uint32_t> mStatus;

  ClassDefNV(StatusMap, 1);
};

/** convert a pair {StatusMap,mask} to a map deid->[vector of bad channel's padids]
 * where the meaning of bad is defined by the mask.
 *
 * Note that if the mask==0 then the output map is empty by construction
 * (as all pads are considered good then).
 *
 * Note also that we do make sure that the output map only contains
 * actually connected pads
 *
 */
std::map<int, std::vector<int>> applyMask(const o2::mch::StatusMap& statusMap, uint32_t mask);

} // namespace o2::mch

#endif
