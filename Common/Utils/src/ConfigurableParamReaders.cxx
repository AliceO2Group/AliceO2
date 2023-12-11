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

#include "CommonUtils/ConfigurableParamReaders.h"
#include "CommonUtils/StringUtils.h"
#include <fairlogger/Logger.h>
#include <filesystem>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/json_parser.hpp>

namespace o2::conf
{
// ------------------------------------------------------------------

boost::property_tree::ptree ConfigurableParamReaders::readINI(std::string const& filepath)
{
  boost::property_tree::ptree pt;
  try {
    boost::property_tree::read_ini(filepath, pt);
  } catch (const boost::property_tree::ptree_error& e) {
    LOG(fatal) << "Failed to read INI config file " << filepath << " (" << e.what() << ")";
  } catch (...) {
    LOG(fatal) << "Unknown error when reading INI config file ";
  }

  return pt;
}

// ------------------------------------------------------------------

boost::property_tree::ptree ConfigurableParamReaders::readJSON(std::string const& filepath)
{
  boost::property_tree::ptree pt;

  try {
    boost::property_tree::read_json(filepath, pt);
  } catch (const boost::property_tree::ptree_error& e) {
    LOG(fatal) << "Failed to read JSON config file " << filepath << " (" << e.what() << ")";
  }

  return pt;
}

boost::property_tree::ptree ConfigurableParamReaders::readConfigFile(std::string const& filepath)
{
  auto inpfilename = o2::utils::Str::concat_string(sInputDir, filepath);
  if (!std::filesystem::exists(inpfilename)) {
    LOG(fatal) << inpfilename << " : config file does not exist!";
  }

  boost::property_tree::ptree pt;

  if (boost::iends_with(inpfilename, ".ini")) {
    pt = ConfigurableParamReaders::readINI(inpfilename);
  } else if (boost::iends_with(inpfilename, ".json")) {
    pt = ConfigurableParamReaders::readJSON(inpfilename);
  } else {
    LOG(fatal) << "Configuration file must have either .ini or .json extension";
  }

  return pt;
}

std::string ConfigurableParamReaders::sInputDir = "";

} // namespace o2::conf
