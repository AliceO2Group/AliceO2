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
#include "MCHRawElecMap/CruLinkId.h"
#include <fmt/format.h>
#include <array>
#include <gsl/span>

namespace o2::mch::raw
{

extern std::array<int, 156> deIdsForAllMCH;

/**@name Mapper templates.

  Those creator functions return functions that can do the mapping to/from 
  DsElecId to DsDetId and to/from CruLinkId to solarId.
    */
///@{

/// From (solarId,groupdId,index) to (deId,dsId)
/// timestamp is foreseen to specify a data taking period (not used for the moment)
/// use 0 to get the latest mapping
template <typename T>
std::function<std::optional<DsDetId>(DsElecId)> createElec2DetMapper(gsl::span<int> deIds,
                                                                     uint64_t timestamp = 0);

/// From (deId,dsId) to (solarId,groupId,index) for a given list of detection elements
template <typename T>
std::function<std::optional<DsElecId>(DsDetId id)> createDet2ElecMapper(gsl::span<int> deIds);

/// From (deId,dsId) to (solarId,groupId,index) for all detection elements
template <typename T>
std::function<std::optional<DsElecId>(DsDetId id)> createDet2ElecMapper()
{
  return createDet2ElecMapper<T>(deIdsForAllMCH);
}

/// From (cruId,linkId) to solarId
template <typename T>
std::function<std::optional<uint16_t>(CruLinkId id)> createCruLink2SolarMapper();

/// From solarId to (cruId,linkId)
template <typename T>
std::function<std::optional<CruLinkId>(uint16_t solarId)> createSolar2CruLinkMapper();
///@}

/**@name Actual mapper types.
    */
///@{

struct ElectronicMapperDummy {
};
struct ElectronicMapperGenerated {
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
