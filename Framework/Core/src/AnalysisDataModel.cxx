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
#include "Framework/Logger.h"
#include <boost/algorithm/string.hpp>

namespace o2
{
namespace aod
{
namespace datamodel
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
static auto columnNamesTrait(framework::pack<C...>)
{
  return std::vector<std::string>{C::columnLabel()...};
}

std::vector<std::string> getColumnNames(header::DataHeader dh)
{
  std::string description = std::string(dh.dataDescription.str);
  std::string origin = std::string(dh.dataOrigin.str);

  // get coulmn names
  // AOD
  if (origin == "AOD") {
    if (description == "TRACK:PAR") {
      return columnNamesTrait(decltype(StoredTracksMetadata{})::table_t::persistent_columns_t{});
    } else if (description == "TRACK:PARCOV") {
      return columnNamesTrait(decltype(StoredTracksCovMetadata{})::table_t::persistent_columns_t{});
    } else if (description == "TRACK:EXTRA") {
      return columnNamesTrait(decltype(TracksExtraMetadata{})::table_t::persistent_columns_t{});
    } else if (description == "CALO") {
      return columnNamesTrait(decltype(CalosMetadata{})::table_t::persistent_columns_t{});
    } else if (description == "CALOTRIGGER") {
      return columnNamesTrait(decltype(CaloTriggersMetadata{})::table_t::persistent_columns_t{});
    } else if (description == "MUON") {
      return columnNamesTrait(decltype(MuonsMetadata{})::table_t::persistent_columns_t{});
    } else if (description == "MUONCLUSTER") {
      return columnNamesTrait(decltype(MuonClustersMetadata{})::table_t::persistent_columns_t{});
    } else if (description == "ZDC") {
      return columnNamesTrait(decltype(ZdcsMetadata{})::table_t::persistent_columns_t{});
    } else if (description == "BC") {
      return columnNamesTrait(decltype(BCsMetadata{})::table_t::persistent_columns_t{});
    } else if (description == "COLLISION") {
      return columnNamesTrait(decltype(CollisionsMetadata{})::table_t::persistent_columns_t{});
    } else if (description == "FT0") {
      return columnNamesTrait(decltype(FT0sMetadata{})::table_t::persistent_columns_t{});
    } else if (description == "FV0") {
      return columnNamesTrait(decltype(FV0sMetadata{})::table_t::persistent_columns_t{});
    } else if (description == "FDD") {
      return columnNamesTrait(decltype(FDDsMetadata{})::table_t::persistent_columns_t{});
    } else if (description == "UNASSIGNEDTRACK") {
      return columnNamesTrait(decltype(UnassignedTracksMetadata{})::table_t::persistent_columns_t{});
    } else if (description == "MCCOLLISION") {
      return columnNamesTrait(decltype(McCollisionsMetadata{})::table_t::persistent_columns_t{});
    } else if (description == "MCTRACKLABEL") {
      return columnNamesTrait(decltype(McTrackLabelsMetadata{})::table_t::persistent_columns_t{});
    } else if (description == "MCCALOLABEL") {
      return columnNamesTrait(decltype(McCaloLabelsMetadata{})::table_t::persistent_columns_t{});
    } else if (description == "MCCOLLISLABEL") {
      return columnNamesTrait(decltype(McCollisionLabelsMetadata{})::table_t::persistent_columns_t{});
    } else if (description == "MCPARTICLE") {
      return columnNamesTrait(decltype(McParticlesMetadata{})::table_t::persistent_columns_t{});
    }
  }

  // RN2
  if (origin == "RN2") {
    if (description == "RUN2V0") {
      return columnNamesTrait(decltype(Run2V0sMetadata{})::table_t::persistent_columns_t{});
    }
  }

  // default: column names = {}
  return std::vector<std::string>({});
}

} // namespace datamodel
} // namespace aod
} // namespace o2
