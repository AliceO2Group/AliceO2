// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "AnalysisDataModelHelpers.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/StringHelpers.h"
#include "Framework/Logger.h"
#include <boost/algorithm/string.hpp>

namespace o2::aod::datamodel
{
std::string getTreeName(header::DataHeader dh)
{
  std::string description = std::string(dh.dataDescription.str);
  std::string origin = std::string(dh.dataOrigin.str);

  // lower case of first part of description
  auto found = description.find_first_of(":");
  std::string treeName = boost::algorithm::to_lower_copy(description).substr(0, found);

  // add prefix according to origin
  if (origin == "AOD") {
    treeName = "O2" + treeName;
  } else if (origin == "RN2") {
    treeName = "Run2" + treeName;
  }

  // exceptions from this
  if (origin == "AOD" && description == "MCCOLLISLABEL") {
    treeName = "O2mccollisionlabel";
  }

  return treeName;
}

template <typename... C>
static constexpr auto columnNamesTrait(framework::pack<C...>)
{
  return std::vector<std::string>{C::columnLabel()...};
}

std::vector<std::string> getColumnNames(header::DataHeader dh)
{
  auto description = std::string(dh.dataDescription.str);
  auto origin = std::string(dh.dataOrigin.str);

  // get column names
  // AOD / RN2
  if (origin == "AOD") {
    if (description == "TRACK:PAR") {
      return columnNamesTrait(typename StoredTracksMetadata::table_t::persistent_columns_t{});
    } else if (description == "TRACK:PARCOV") {
      return columnNamesTrait(typename StoredTracksCovMetadata::table_t::persistent_columns_t{});
    } else if (description == "TRACK:EXTRA") {
      return columnNamesTrait(typename TracksExtraMetadata::table_t::persistent_columns_t{});
    }
  }

  // default: column names = {}
  return std::vector<std::string>({});
}

} // namespace o2::aod::datamodel
