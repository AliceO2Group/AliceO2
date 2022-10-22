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

#ifndef O2_MCH_RAW_ELECMAP_MAPPER_H
#define O2_MCH_RAW_ELECMAP_MAPPER_H

#include "MCHRawElecMap/DsDetId.h"
#include "MCHRawElecMap/DsElecId.h"
#include "MCHRawElecMap/FeeLinkId.h"
#include <array>
#include <cstdint>
#include <fmt/format.h>
#include <functional>
#include <gsl/span>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace o2::mch::raw
{

/**@name Mapper templates.

  Those creator functions return functions that can do the mapping to/from
  DsElecId to DsDetId and to/from FeeLinkId to solarId.
    */
///@{

/// From (solarId,groupdId,index) to (deId,dsId)
/// timestamp is foreseen to specify a data taking period (not used for the moment)
/// use 0 to get the latest mapping
using Elec2DetMapper = std::function<std::optional<DsDetId>(DsElecId)>;
template <typename T>
Elec2DetMapper createElec2DetMapper(uint64_t timestamp = 0);

/// From (deId,dsId) to (solarId,groupId,index)
using Det2ElecMapper = std::function<std::optional<DsElecId>(DsDetId id)>;
template <typename T>
Det2ElecMapper createDet2ElecMapper();

/// From (feeId,linkId) to solarId
using FeeLink2SolarMapper = std::function<std::optional<uint16_t>(FeeLinkId id)>;
template <typename T>
FeeLink2SolarMapper createFeeLink2SolarMapper();

/// From solarId to (feeId,linkId)
using Solar2FeeLinkMapper = std::function<std::optional<FeeLinkId>(uint16_t solarId)>;
template <typename T>
Solar2FeeLinkMapper createSolar2FeeLinkMapper();
///@}

/// List of Solar Unique Ids for a given detection element id
template <typename T>
std::set<uint16_t> getSolarUIDs(int deid);

/// List of Solar Unique Ids for all MCH
template <typename T>
std::set<uint16_t> getSolarUIDs();

/// List of Solar Unique Ids for a given FeeId
template <typename T>
std::set<uint16_t> getSolarUIDsPerFeeId(uint16_t feeId);

/// List of Dual Sampa handled by a given Solar
template <typename T>
std::set<DsDetId> getDualSampas(uint16_t solarId);

/// List of Dual Sampa handled by a given FeeId
template <typename T>
std::set<DsDetId> getDualSampasPerFeeId(uint16_t feeId);

/**@name Actual mapper types.
 */
///@{

struct ElectronicMapperDummy {
};
struct ElectronicMapperGenerated {
};
struct ElectronicMapperString {
  static std::string sCruMap;
  static std::string sFecMap;
};
///@}

/** Return the full set of Dual Sampa Electronic Id of MCH,
 * for a given electronic mapping */
template <typename T>
std::set<DsElecId> getAllDs();

extern std::array<int, 2> deIdsOfCH1R;
extern std::array<int, 2> deIdsOfCH1L;
extern std::array<int, 2> deIdsOfCH2R;
extern std::array<int, 2> deIdsOfCH2L;
extern std::array<int, 2> deIdsOfCH3R;
extern std::array<int, 2> deIdsOfCH3L;
extern std::array<int, 2> deIdsOfCH4R;
extern std::array<int, 2> deIdsOfCH4L;
extern std::array<int, 9> deIdsOfCH5R;
extern std::array<int, 9> deIdsOfCH5L;
extern std::array<int, 9> deIdsOfCH6R;
extern std::array<int, 9> deIdsOfCH6L;
extern std::array<int, 13> deIdsOfCH7R;
extern std::array<int, 13> deIdsOfCH7L;
extern std::array<int, 13> deIdsOfCH8R;
extern std::array<int, 13> deIdsOfCH8L;
extern std::array<int, 13> deIdsOfCH9R;
extern std::array<int, 13> deIdsOfCH9L;
extern std::array<int, 13> deIdsOfCH10R;
extern std::array<int, 13> deIdsOfCH10L;

// test whether all solars have a corresponding FeeLinkId
// and the reverse as well.
// @returns vector of error messages. If empty the check is ok
template <typename T>
std::vector<std::string> solar2FeeLinkConsistencyCheck();

} // namespace o2::mch::raw

#endif
