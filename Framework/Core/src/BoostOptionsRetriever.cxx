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
#include <boost/program_options/options_description.hpp>

#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>

using namespace o2::framework;
namespace bpo = boost::program_options;

namespace o2::framework
{

BoostOptionsRetriever::BoostOptionsRetriever(bool ignoreUnknown,
                                             int argc, char** argv)
  : mDescription{std::make_unique<boost::program_options::options_description>("ALICE O2 Framework - Available options")},
    mArgc{argc},
    mArgv{argv},
    mIgnoreUnknown{ignoreUnknown}
{
}

void BoostOptionsRetriever::update(std::vector<ConfigParamSpec> const& specs,
                                   boost::property_tree::ptree& store,
                                   boost::property_tree::ptree& provenance)
{
  auto options = mDescription->add_options();
  for (const auto& spec : specs) {
    const char* name = spec.name.c_str();
    const char* help = spec.help.c_str();
    // FIXME: propagate default value?
    switch (spec.type) {
      case VariantType::Int:
        options = options(name, bpo::value<int>()->default_value(spec.defaultValue.get<int>()), help);
        break;
      case VariantType::Int64:
        options = options(name, bpo::value<int64_t>()->default_value(spec.defaultValue.get<int64_t>()), help);
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
      case VariantType::ArrayInt:
      case VariantType::ArrayFloat:
      case VariantType::ArrayDouble:
      case VariantType::ArrayBool:
      case VariantType::ArrayString:
      case VariantType::Array2DInt:
      case VariantType::Array2DFloat:
      case VariantType::Array2DDouble:
        options = options(name, bpo::value<std::string>()->default_value(spec.defaultValue.asString()), help);
        break;
      case VariantType::LabeledArrayInt:
      case VariantType::LabeledArrayFloat:
      case VariantType::LabeledArrayDouble:
      case VariantType::Unknown:
      case VariantType::Empty:
        break;
    };
  }

  using namespace bpo::command_line_style;
  auto style = (allow_short | short_allow_adjacent | short_allow_next | allow_long | long_allow_adjacent | long_allow_next | allow_sticky | allow_dash_for_short);

  auto parsed = mIgnoreUnknown ? bpo::command_line_parser(mArgc, mArgv).options(*mDescription).style(style).allow_unregistered().run()
                               : bpo::parse_command_line(mArgc, mArgv, *mDescription, style);
  bpo::variables_map vmap;
  bpo::store(parsed, vmap);
  PropertyTreeHelpers::populate(specs, store, vmap, provenance);
}

} // namespace o2::framework
