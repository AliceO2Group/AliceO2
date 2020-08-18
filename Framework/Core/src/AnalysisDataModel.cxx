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

namespace o2
{
namespace aod
{
namespace datamodel
{
std::string getTreeName(header::DataHeader dh)
{
  std::string tableName = std::string(dh.dataDescription.str);

  // convert table name to tree name
  // default: tree name = table name
  if (tableName == "TRACKPAR") {
    return decltype(StoredTracksMetadata{})::mTreeName;
  } else if (tableName == "TRACKPARCOV") {
    return decltype(StoredTracksCovMetadata{})::mTreeName;
  } else if (tableName == "TRACKEXTRA") {
    return decltype(TracksExtraMetadata{})::mTreeName;
  } else if (tableName == "CALO") {
    return decltype(CalosMetadata{})::mTreeName;
  } else if (tableName == "CALOTRIGGER") {
    return decltype(CaloTriggersMetadata{})::mTreeName;
  } else if (tableName == "MUON") {
    return decltype(MuonsMetadata{})::mTreeName;
  } else if (tableName == "MUONCLUSTER") {
    return decltype(MuonClustersMetadata{})::mTreeName;
  } else if (tableName == "ZDC") {
    return decltype(ZdcsMetadata{})::mTreeName;
  } else if (tableName == "BC") {
    return decltype(BCsMetadata{})::mTreeName;
  } else if (tableName == "COLLISION") {
    return decltype(CollisionsMetadata{})::mTreeName;
  } else if (tableName == "FT0") {
    return decltype(FT0sMetadata{})::mTreeName;
  } else if (tableName == "FV0") {
    return decltype(FV0sMetadata{})::mTreeName;
  } else if (tableName == "FDD") {
    return decltype(FDDsMetadata{})::mTreeName;
  } else if (tableName == "UNASSIGNEDTRACK") {
    return decltype(UnassignedTracksMetadata{})::mTreeName;
  } else if (tableName == "RUN2V0") {
    return decltype(Run2V0sMetadata{})::mTreeName;
  } else if (tableName == "MCCOLLISION") {
    return decltype(McCollisionsMetadata{})::mTreeName;
  } else if (tableName == "MCTRACKLABEL") {
    return decltype(McTrackLabelsMetadata{})::mTreeName;
  } else if (tableName == "MCCALOLABEL") {
    return decltype(McCaloLabelsMetadata{})::mTreeName;
  } else if (tableName == "MCCOLLISLABEL") {
    return decltype(McCollisionLabelsMetadata{})::mTreeName;
  } else if (tableName == "MCPARTICLE") {
    return decltype(McParticlesMetadata{})::mTreeName;
  } else {
    return tableName;
  }
}

template <typename... C>
static auto columnNamesTrait(framework::pack<C...>)
{
  return std::vector<std::string>{C::columnLabel()...};
}

std::vector<std::string> getColumnNames(std::string tableName)
{
  // get coulmn names
  // default: column names = {}
  if (tableName == "TRACKPAR") {
    return columnNamesTrait(decltype(StoredTracksMetadata{})::table_t::persistent_columns_t{});
  } else if (tableName == "TRACKPARCOV") {
    return columnNamesTrait(decltype(StoredTracksCovMetadata{})::table_t::persistent_columns_t{});
  } else if (tableName == "TRACKEXTRA") {
    return columnNamesTrait(decltype(TracksExtraMetadata{})::table_t::persistent_columns_t{});
  } else if (tableName == "CALO") {
    return columnNamesTrait(decltype(CalosMetadata{})::table_t::persistent_columns_t{});
  } else if (tableName == "CALOTRIGGER") {
    return columnNamesTrait(decltype(CaloTriggersMetadata{})::table_t::persistent_columns_t{});
  } else if (tableName == "MUON") {
    return columnNamesTrait(decltype(MuonsMetadata{})::table_t::persistent_columns_t{});
  } else if (tableName == "MUONCLUSTER") {
    return columnNamesTrait(decltype(MuonClustersMetadata{})::table_t::persistent_columns_t{});
  } else if (tableName == "ZDC") {
    return columnNamesTrait(decltype(ZdcsMetadata{})::table_t::persistent_columns_t{});
  } else if (tableName == "BC") {
    return columnNamesTrait(decltype(BCsMetadata{})::table_t::persistent_columns_t{});
  } else if (tableName == "COLLISION") {
    return columnNamesTrait(decltype(CollisionsMetadata{})::table_t::persistent_columns_t{});
  } else if (tableName == "FT0") {
    return columnNamesTrait(decltype(FT0sMetadata{})::table_t::persistent_columns_t{});
  } else if (tableName == "FV0") {
    return columnNamesTrait(decltype(FV0sMetadata{})::table_t::persistent_columns_t{});
  } else if (tableName == "FDD") {
    return columnNamesTrait(decltype(FDDsMetadata{})::table_t::persistent_columns_t{});
  } else if (tableName == "UNASSIGNEDTRACK") {
    return columnNamesTrait(decltype(UnassignedTracksMetadata{})::table_t::persistent_columns_t{});
  } else if (tableName == "RUN2V0") {
    return columnNamesTrait(decltype(Run2V0sMetadata{})::table_t::persistent_columns_t{});
  } else if (tableName == "MCCOLLISION") {
    return columnNamesTrait(decltype(McCollisionsMetadata{})::table_t::persistent_columns_t{});
  } else if (tableName == "MCTRACKLABEL") {
    return columnNamesTrait(decltype(McTrackLabelsMetadata{})::table_t::persistent_columns_t{});
  } else if (tableName == "MCCALOLABEL") {
    return columnNamesTrait(decltype(McCaloLabelsMetadata{})::table_t::persistent_columns_t{});
  } else if (tableName == "MCCOLLISLABEL") {
    return columnNamesTrait(decltype(McCollisionLabelsMetadata{})::table_t::persistent_columns_t{});
  } else if (tableName == "MCPARTICLE") {
    return columnNamesTrait(decltype(McParticlesMetadata{})::table_t::persistent_columns_t{});
  } else {
    return std::vector<std::string>({});
  }
}

} // namespace datamodel
} // namespace aod
} // namespace o2
