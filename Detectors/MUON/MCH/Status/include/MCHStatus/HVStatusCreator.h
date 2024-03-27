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

#ifndef O2_MCH_HV_STATUS_CREATOR_H_
#define O2_MCH_HV_STATUS_CREATOR_H_

#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"

namespace o2::mch
{

class StatusMap;

/**
 * @class HVStatusCreator
 * @brief Find HV issues from DCS data points and add them to the status map
 *
 * This is a 3 step procedure:
 *
 * 1) Find all potential issues from the DCS data points stored in one HV file.
 * This must be done each time a new HV file is read from the CCDB. It stores in
 * an internal map the time range(s) of the issue(s) for each affected HV channel.
 *
 * 2) Find all real issues at a given time stamp.
 * This must be done for every TF. It updates the internal list of bad HV
 * channels if it is different from the current one and tells if that happens.
 *
 * 3) Update the status maps if needed.
 * This must be done each time the current list of bad HV channel has changed.
 * It adds every electronics channels associated to the bad HV channels into the
 * status map given as a parameter.
 */
class HVStatusCreator
{
 public:
  using DPID = dcs::DataPointIdentifier;
  using DPVAL = dcs::DataPointValue;
  using DPMAP = std::unordered_map<DPID, std::vector<DPVAL>>;

  /**
   * Find all HV issues and their time ranges
   * @param dpMap DCS HV data points from CCDB
   */
  void findBadHVs(const DPMAP& dpMap);

  /**
   * Find HV issues at a given time stamp
   * @param timestamp time stamp of interest
   * @return true if the list of issues has changed
   */
  bool findCurrentBadHVs(uint64_t timestamp);

  /**
   * Add channels affected by current HV issues to the status map
   * @param statusMap statusMap to update
   */
  void updateStatusMap(StatusMap& statusMap);

  /**
   * clear the internal lists of HV issues
   */
  void clear()
  {
    mBadHVTimeRanges.clear();
    mCurrentBadHVs.clear();
  }

 private:
  /// @brief internal structure to define a time range
  struct TimeRange {
    uint64_t begin = 0; ///< beginning of time range
    uint64_t end = 0;   ///< end of time range

    /**
     * @brief check if the time range contains the given time stamp
     * @param timestamp time stamp of interest
     * @return true if the time stamp is in the time range
     */
    bool contains(uint64_t timestamp) const { return timestamp >= begin && timestamp < end; }
  };

  /// map of bad HV channels with the time ranges concerned
  std::unordered_map<std::string, std::vector<TimeRange>> mBadHVTimeRanges{};
  std::set<std::string> mCurrentBadHVs{}; ///< current list of bad HV channels
};

} // namespace o2::mch

#endif // O2_MCH_HV_STATUS_CREATOR_H_
