// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_ELECMAP_MAPPER_H
#define O2_MCH_RAW_ELECMAP_MAPPER_H

#include <functional>
#include <optional>
#include <set>
#include <stdexcept>
#include <cstdint>
#include "MCHRawElecMap/DsDetId.h"
#include "MCHRawElecMap/DsElecId.h"
#include "MCHRawElecMap/FeeLinkId.h"
#include <fmt/format.h>
#include <array>
#include <gsl/span>

namespace o2::mch::raw
{

extern std::array<int, 156> deIdsForAllMCH;

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

extern std::array<int, 9> deIdsOfCH5R;
extern std::array<int, 9> deIdsOfCH5L;
extern std::array<int, 9> deIdsOfCH6R;
extern std::array<int, 9> deIdsOfCH6L;
extern std::array<int, 13> deIdsOfCH7R;
extern std::array<int, 13> deIdsOfCH7L;
extern std::array<int, 13> deIdsOfCH8R;
extern std::array<int, 13> deIdsOfCH8L;

} // namespace o2::mch::raw

#endif
