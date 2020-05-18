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

#include "Configuration/ConfigurationInterface.h"
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

ConfigurationOptionsRetriever::ConfigurationOptionsRetriever(ConfigurationInterface* cfg,
                                                             std::string const& mainKey)
  : mCfg{cfg},
    mMainKey{mainKey}
{
}

void ConfigurationOptionsRetriever::update(std::vector<ConfigParamSpec> const& schema,
                                           boost::property_tree::ptree& store,
                                           boost::property_tree::ptree& provenance)
{
  boost::property_tree::ptree in;
  try {
    in = mCfg->getRecursive(mMainKey);
  } catch (...) {
    in.clear();
  }
  PropertyTreeHelpers::populate(schema, store, in, provenance, "configuration:" + mMainKey);
}

} // namespace o2::framework
