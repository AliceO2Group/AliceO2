// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "ConfigurationOptionsRetriever.h"

#include "Framework/ConfigParamSpec.h"
#include "PropertyTreeHelpers.h"

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <string>
#include <vector>
#include <iostream>

using namespace o2::framework;
using namespace o2::configuration;
namespace bpo = boost::program_options;
namespace bpt = boost::property_tree;

namespace o2::framework
{

ConfigurationOptionsRetriever::ConfigurationOptionsRetriever(std::vector<ConfigParamSpec> const& schema,
                                                             ConfigurationInterface* cfg,
                                                             std::string const& mainKey)
  : mCfg{cfg},
    mStore{}
{
  PropertyTreeHelpers::populate(schema, mStore, cfg->getRecursive(mainKey));
}

bool ConfigurationOptionsRetriever::isSet(const char* key) const
{
  return (mStore.count(key) > 0);
}

int ConfigurationOptionsRetriever::getInt(const char* key) const
{
  return mStore.get<int>(key);
}

int64_t ConfigurationOptionsRetriever::getInt64(const char* key) const
{
  return mStore.get<int64_t>(key);
}

float ConfigurationOptionsRetriever::getFloat(const char* key) const
{
  return mStore.get<float>(key);
}

double ConfigurationOptionsRetriever::getDouble(const char* key) const
{
  return mStore.get<double>(key);
}

bool ConfigurationOptionsRetriever::getBool(const char* key) const
{
  return mStore.get<bool>(key);
}

std::string ConfigurationOptionsRetriever::getString(const char* key) const
{
  return mStore.get<std::string>(key);
}

boost::property_tree::ptree ConfigurationOptionsRetriever::getPTree(const char* key) const
{
  return mStore.get_child(key);
}

} // namespace o2::framework
