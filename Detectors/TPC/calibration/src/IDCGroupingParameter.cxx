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

#include "TPCCalibration/IDCGroupingParameter.h"
#include "Framework/Logger.h"
#include "Algorithm/RangeTokenizer.h"
#include <boost/property_tree/ptree.hpp>
#include <cassert>
#include <stdexcept>

using namespace o2::tpc;
O2ParamImpl(o2::tpc::ParameterIDCGroup);
O2ParamImpl(o2::tpc::ParameterIDCCompression);

void o2::tpc::ParameterIDCGroup::setGroupingParameterFromString(const std::string sgroupPads, const std::string sgroupRows, const std::string sgroupLastRowsThreshold, const std::string sgroupLastPadsThreshold)
{
  auto vgroupPads = o2::RangeTokenizer::tokenize<int>(sgroupPads);
  auto vgroupRows = o2::RangeTokenizer::tokenize<int>(sgroupRows);
  auto vgroupLastRowsThreshold = o2::RangeTokenizer::tokenize<int>(sgroupLastRowsThreshold);
  auto vgroupLastPadsThreshold = o2::RangeTokenizer::tokenize<int>(sgroupLastPadsThreshold);

  if (vgroupPads.size() == 1) {
    vgroupPads = std::vector<int>(Mapper::NREGIONS, vgroupPads.front());
  } else if (vgroupPads.size() != Mapper::NREGIONS) {
    LOGP(error, "wrong number of parameters inserted for groupPads (n={}). Number should be 1 or {}", vgroupPads.size(), Mapper::NREGIONS);
  }

  if (vgroupRows.size() == 1) {
    vgroupRows = std::vector<int>(Mapper::NREGIONS, vgroupRows.front());
  } else if (vgroupRows.size() != Mapper::NREGIONS) {
    LOGP(error, "wrong number of parameters inserted for groupRows (n={}). Number should be 1 or {}", vgroupRows.size(), Mapper::NREGIONS);
  }

  if (vgroupLastRowsThreshold.size() == 1) {
    vgroupLastRowsThreshold = std::vector<int>(Mapper::NREGIONS, vgroupLastRowsThreshold.front());
  } else if (vgroupLastRowsThreshold.size() != Mapper::NREGIONS) {
    LOGP(error, "wrong number of parameters inserted for groupLastRowsThreshold (n={}). Number should be 1 or {}", vgroupLastRowsThreshold.size(), Mapper::NREGIONS);
  }

  if (vgroupLastPadsThreshold.size() == 1) {
    vgroupLastPadsThreshold = std::vector<int>(Mapper::NREGIONS, vgroupLastPadsThreshold.front());
  } else if (vgroupLastPadsThreshold.size() != Mapper::NREGIONS) {
    LOGP(error, "wrong number of parameters inserted for groupLastPadsThreshold (n={}). Number should be 1 or {}", vgroupLastPadsThreshold.size(), Mapper::NREGIONS);
  }

  for (int i = 0; i < Mapper::NREGIONS; ++i) {
    o2::conf::ConfigurableParam::setValue<unsigned char>("TPCIDCGroupParam", fmt::format("groupPads[{}]", i).data(), vgroupPads[i]);
    o2::conf::ConfigurableParam::setValue<unsigned char>("TPCIDCGroupParam", fmt::format("groupRows[{}]", i).data(), vgroupRows[i]);
    o2::conf::ConfigurableParam::setValue<unsigned char>("TPCIDCGroupParam", fmt::format("groupLastRowsThreshold[{}]", i).data(), vgroupLastRowsThreshold[i]);
    o2::conf::ConfigurableParam::setValue<unsigned char>("TPCIDCGroupParam", fmt::format("groupLastPadsThreshold[{}]", i).data(), vgroupLastPadsThreshold[i]);
  }
}

unsigned int o2::tpc::ParameterIDCGroup::getTotalGroupPadsSectorEdgesHelper(unsigned int groupPadsSectorEdges)
{
  if (groupPadsSectorEdges == 0) {
    return 0;
  }
  return (groupPadsSectorEdges % 10) + getTotalGroupPadsSectorEdgesHelper(groupPadsSectorEdges / 10);
}

unsigned int o2::tpc::ParameterIDCGroup::getGroupedPadsSectorEdgesHelper(unsigned int groupPadsSectorEdges)
{
  if (groupPadsSectorEdges == 0) {
    return 0;
  }
  return 1 + getGroupedPadsSectorEdgesHelper(groupPadsSectorEdges / 10);
}

EdgePadGroupingMethod o2::tpc::ParameterIDCGroup::getEdgePadGroupingType(unsigned int groupPadsSectorEdges)
{
  const auto groupPadsSectorEdgesTmp = groupPadsSectorEdges % 10;
  switch (groupPadsSectorEdgesTmp) {
    case static_cast<unsigned int>(EdgePadGroupingMethod::NO):
    case static_cast<unsigned int>(EdgePadGroupingMethod::ROWS):
      return static_cast<EdgePadGroupingMethod>(groupPadsSectorEdgesTmp);
    default:
      throw std::invalid_argument("Wrong type for EdgePadGroupingMethod can either be NO or ROWS");
  }
}

unsigned int o2::tpc::ParameterIDCGroup::getPadsInGroupSectorEdgesHelper(unsigned int groupPadsSectorEdges, const unsigned int group)
{
  return groupPadsSectorEdges / static_cast<int>(std::pow(10, group)) % 10;
}
