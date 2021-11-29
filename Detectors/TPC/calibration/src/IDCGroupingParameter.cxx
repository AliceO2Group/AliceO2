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
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>

using namespace o2::tpc;
O2ParamImpl(o2::tpc::ParameterIDCGroup);
O2ParamImpl(o2::tpc::ParameterIDCCompression);

void o2::tpc::ParameterIDCGroup::setGroupingParameterFromString(const std::string sgroupPads, const std::string sgroupRows, const std::string sgroupLastRowsThreshold, const std::string sgroupLastPadsThreshold)
{
  // convert input string to std::vector<unsigned char>
  const boost::char_separator<char> sep(","); /// char separator for the tokenizer
  const boost::tokenizer<boost::char_separator<char>> tgroupPads(sgroupPads, sep);
  const boost::tokenizer<boost::char_separator<char>> tgroupRows(sgroupRows, sep);
  const boost::tokenizer<boost::char_separator<char>> tgroupLastRowsThreshold(sgroupLastRowsThreshold, sep);
  const boost::tokenizer<boost::char_separator<char>> tgroupLastPadsThreshold(sgroupLastPadsThreshold, sep);

  std::vector<unsigned char> vgroupPads;
  std::vector<unsigned char> vgroupRows;
  std::vector<unsigned char> vgroupLastRowsThreshold;
  std::vector<unsigned char> vgroupLastPadsThreshold;
  vgroupPads.reserve(Mapper::NREGIONS);
  vgroupRows.reserve(Mapper::NREGIONS);
  vgroupLastRowsThreshold.reserve(Mapper::NREGIONS);
  vgroupLastPadsThreshold.reserve(Mapper::NREGIONS);

  std::transform(tgroupPads.begin(), tgroupPads.end(), std::back_inserter(vgroupPads), &boost::lexical_cast<int, std::string>);
  std::transform(tgroupRows.begin(), tgroupRows.end(), std::back_inserter(vgroupRows), &boost::lexical_cast<int, std::string>);
  std::transform(tgroupLastRowsThreshold.begin(), tgroupLastRowsThreshold.end(), std::back_inserter(vgroupLastRowsThreshold), &boost::lexical_cast<int, std::string>);
  std::transform(tgroupLastPadsThreshold.begin(), tgroupLastPadsThreshold.end(), std::back_inserter(vgroupLastPadsThreshold), &boost::lexical_cast<int, std::string>);

  if (vgroupPads.size() == 1) {
    vgroupPads = std::vector<unsigned char>(Mapper::NREGIONS, vgroupPads.front());
  } else if (vgroupPads.size() != Mapper::NREGIONS) {
    LOGP(error, "wrong number of parameters inserted for groupPads (n={}). Number should be 1 or {}", vgroupPads.size(), Mapper::NREGIONS);
  }

  if (vgroupRows.size() == 1) {
    vgroupRows = std::vector<unsigned char>(Mapper::NREGIONS, vgroupRows.front());
  } else if (vgroupRows.size() != Mapper::NREGIONS) {
    LOGP(error, "wrong number of parameters inserted for groupRows (n={}). Number should be 1 or {}", vgroupRows.size(), Mapper::NREGIONS);
  }

  if (vgroupLastRowsThreshold.size() == 1) {
    vgroupLastRowsThreshold = std::vector<unsigned char>(Mapper::NREGIONS, vgroupLastRowsThreshold.front());
  } else if (vgroupLastRowsThreshold.size() != Mapper::NREGIONS) {
    LOGP(error, "wrong number of parameters inserted for groupLastRowsThreshold (n={}). Number should be 1 or {}", vgroupLastRowsThreshold.size(), Mapper::NREGIONS);
  }

  if (vgroupLastPadsThreshold.size() == 1) {
    vgroupLastPadsThreshold = std::vector<unsigned char>(Mapper::NREGIONS, vgroupLastPadsThreshold.front());
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
