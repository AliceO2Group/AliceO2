// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/BoostOptionsRetriever.h"
#include "Framework/ConfigParamSpec.h"

#include "PropertyTreeHelpers.h"

#include <boost/program_options.hpp>

#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>

using namespace o2::framework;
namespace bpo = boost::program_options;

namespace o2
{
namespace framework
{

BoostOptionsRetriever::BoostOptionsRetriever(std::vector<ConfigParamSpec> const& specs,
                                             bool ignoreUnknown,
                                             int& argc, char**& argv)
  : mStore{},
    mDescription{"ALICE O2 Framework - Available options"},
    mIgnoreUnknown{ignoreUnknown}
{
  auto options = mDescription.add_options();
  for (auto& spec : specs) {
    const char* name = spec.name.c_str();
    const char* help = spec.help.c_str();
    // FIXME: propagate default value?
    switch (spec.type) {
      case VariantType::Int:
      case VariantType::Int64:
        options = options(name, bpo::value<int>()->default_value(spec.defaultValue.get<int>()), help);
        break;
      case VariantType::Float:
        options = options(name, bpo::value<float>()->default_value(spec.defaultValue.get<float>()), help);
        break;
      case VariantType::Double:
        options = options(name, bpo::value<double>()->default_value(spec.defaultValue.get<double>()), help);
        break;
      case VariantType::String:
        options = options(name, bpo::value<std::string>()->default_value(spec.defaultValue.get<const char*>()), help);
        break;
      case VariantType::Bool:
        options = options(name, bpo::value<bool>()->zero_tokens()->default_value(spec.defaultValue.get<bool>()), help);
        break;
      case VariantType::Unknown:
      case VariantType::Empty:
        break;
    };
  }

  auto parsed = mIgnoreUnknown ? bpo::command_line_parser(argc, argv).options(mDescription).allow_unregistered().run()
                               : bpo::parse_command_line(argc, argv, mDescription);
  bpo::variables_map vmap;
  bpo::store(parsed, vmap);
  PropertyTreeHelpers::populate(specs, mStore, vmap);
}

int BoostOptionsRetriever::getInt(const char* key) const
{
  return mStore.get<int>(key);
}

float BoostOptionsRetriever::getFloat(const char* key) const
{
  return mStore.get<float>(key);
}

double BoostOptionsRetriever::getDouble(const char* key) const
{
  return mStore.get<double>(key);
}

bool BoostOptionsRetriever::getBool(const char* key) const
{
  return mStore.get<bool>(key);
}

std::string BoostOptionsRetriever::getString(const char* key) const
{
  return mStore.get<std::string>(key);
}

boost::property_tree::ptree BoostOptionsRetriever::getPTree(const char* key) const
{
  return mStore.get_child(key);
}

} // namespace framework
} // namespace o2
