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
      case VariantType::Int8:
        options = options(name, bpo::value<int8_t>()->default_value(spec.defaultValue.get<int8_t>()), help);
        break;
      case VariantType::Int16:
        options = options(name, bpo::value<int16_t>()->default_value(spec.defaultValue.get<int16_t>()), help);
        break;
      case VariantType::UInt8:
        options = options(name, bpo::value<uint8_t>()->default_value(spec.defaultValue.get<uint8_t>()), help);
        break;
      case VariantType::UInt16:
        options = options(name, bpo::value<uint16_t>()->default_value(spec.defaultValue.get<uint16_t>()), help);
        break;
      case VariantType::UInt32:
        options = options(name, bpo::value<uint32_t>()->default_value(spec.defaultValue.get<uint32_t>()), help);
        break;
      case VariantType::UInt64:
        options = options(name, bpo::value<uint64_t>()->default_value(spec.defaultValue.get<uint64_t>()), help);
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
      case VariantType::Dict:
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
