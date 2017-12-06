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
#include <boost/program_options.hpp>
#include <string>
#include <vector>

using namespace o2::framework;
namespace bpo = boost::program_options;

namespace o2 {
namespace framework {

BoostOptionsRetriever::BoostOptionsRetriever(std::vector<ConfigParamSpec> &specs)
: mVariables{},
  mDescription{"ALICE O2 Framework - Available options"}
{
  auto options = mDescription.add_options();
  for (auto & spec : specs) {
    const char *name = spec.name.c_str();
    const char *help = spec.help.c_str();
    // FIXME: propagate default value?
    switch(spec.type) {
      case VariantType::Int:
      case VariantType::Int64:
        options = options(name, bpo::value<int>(), help);
        break;
      case VariantType::Float:
        options = options(name, bpo::value<float>(), help);
        break;
      case VariantType::Double:
        options = options(name, bpo::value<double>(), help);
        break;
      case VariantType::String:
        options = options(name, bpo::value<std::string>(), help);
        break;
      case VariantType::Bool:
        options = options(name, bpo::value<bool>(), help);
        break;
      case VariantType::Unknown:
      case VariantType::Empty:
        break;
    };
  }
}

void BoostOptionsRetriever::parseArgs(int argc, char **argv) {
  bpo::store(parse_command_line(argc, argv, mDescription), mVariables);
  bpo::notify(mVariables);
}

int BoostOptionsRetriever::getInt(const char *key) {
  return mVariables[key].as<int>();
}

float BoostOptionsRetriever::getFloat(const char *key) {
  return mVariables[key].as<float>();
}

double BoostOptionsRetriever::getDouble(const char *key) {
  return mVariables[key].as<double>();
}

bool BoostOptionsRetriever::getBool(const char *key) {
  return mVariables[key].as<bool>();
}

std::string BoostOptionsRetriever::getString(const char *key) {
  return mVariables[key].as<std::string>();
}

std::vector<std::string> BoostOptionsRetriever::getVString(const char *key) {
  return mVariables[key].as<std::vector<std::string>>();
}

} // namespace framework
} // namespace o2
