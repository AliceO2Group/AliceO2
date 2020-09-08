// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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
    } else if (description == "CALO") {
      return columnNamesTrait(typename CalosMetadata::table_t::persistent_columns_t{});
    } else if (description == "CALOTRIGGER") {
      return columnNamesTrait(typename CaloTriggersMetadata::table_t::persistent_columns_t{});
    } else if (description == "MUON") {
      return columnNamesTrait(typename StoredMuonsMetadata::table_t::persistent_columns_t{});
    } else if (description == "MUONCLUSTER") {
      return columnNamesTrait(typename MuonClustersMetadata::table_t::persistent_columns_t{});
    } else if (description == "ZDC") {
      return columnNamesTrait(typename ZdcsMetadata::table_t::persistent_columns_t{});
    } else if (description == "BC") {
      return columnNamesTrait(typename BCsMetadata::table_t::persistent_columns_t{});
    } else if (description == "COLLISION") {
      return columnNamesTrait(typename CollisionsMetadata::table_t::persistent_columns_t{});
    } else if (description == "FT0") {
      return columnNamesTrait(typename FT0sMetadata::table_t::persistent_columns_t{});
    } else if (description == "FV0") {
      return columnNamesTrait(typename FV0sMetadata::table_t::persistent_columns_t{});
    } else if (description == "FDD") {
      return columnNamesTrait(typename FDDsMetadata::table_t::persistent_columns_t{});
    } else if (description == "UNASSIGNEDTRACK") {
      return columnNamesTrait(typename UnassignedTracksMetadata::table_t::persistent_columns_t{});
    } else if (description == "MCCOLLISION") {
      return columnNamesTrait(typename McCollisionsMetadata::table_t::persistent_columns_t{});
    } else if (description == "MCTRACKLABEL") {
      return columnNamesTrait(typename McTrackLabelsMetadata::table_t::persistent_columns_t{});
    } else if (description == "MCCALOLABEL") {
      return columnNamesTrait(typename McCaloLabelsMetadata::table_t::persistent_columns_t{});
    } else if (description == "MCCOLLISLABEL") {
      return columnNamesTrait(typename McCollisionLabelsMetadata::table_t::persistent_columns_t{});
    } else if (description == "MCPARTICLE") {
      return columnNamesTrait(typename McParticlesMetadata::table_t::persistent_columns_t{});
    }
  } else if (origin == "RN2") {
    if (description == "V0") {
      return columnNamesTrait(typename Run2V0sMetadata::table_t::persistent_columns_t{});
    }
  }

  // default: column names = {}
  return std::vector<std::string>({});
}

} // namespace datamodel::aod::o2
