// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/FairOptionsRetriever.h"
#include "Framework/ConfigParamSpec.h"
#include "PropertyTreeHelpers.h"

#include <options/FairMQProgOptions.h>
#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <string>
#include <vector>

using namespace o2::framework;
namespace bpo = boost::program_options;
namespace bpt = boost::property_tree;

namespace o2
{
namespace framework
{

FairOptionsRetriever::FairOptionsRetriever(std::vector<ConfigParamSpec> const& schema, const FairMQProgOptions* opts)
  : mOpts{opts},
    mStore{}
{
  PropertyTreeHelpers::populate(schema, mStore, mOpts->GetVarMap());
}

int FairOptionsRetriever::getInt(const char* key) const
{
  return mStore.get<int>(key);
}

float FairOptionsRetriever::getFloat(const char* key) const
{
  return mStore.get<float>(key);
}

double FairOptionsRetriever::getDouble(const char* key) const
{
  return mStore.get<double>(key);
}

bool FairOptionsRetriever::getBool(const char* key) const
{
  return mStore.get<bool>(key);
}

std::string FairOptionsRetriever::getString(const char* key) const
{
  return mStore.get<std::string>(key);
}

boost::property_tree::ptree FairOptionsRetriever::getPTree(const char* key) const
{
  return mStore.get_child(key);
}

} // namespace framework
} // namespace o2
