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
#include "Framework/ConfigParamsHelper.h"
#include "Framework/ConfigParamSpec.h"
#include <boost/program_options.hpp>

#include <string>
#include <vector>
#include <iostream>

namespace bpo = boost::program_options;

namespace o2::framework
{

/// this creates the boost program options description from the ConfigParamSpec
/// taking the VariantType into account
void ConfigParamsHelper::populateBoostProgramOptions(
  bpo::options_description& options,
  const std::vector<ConfigParamSpec>& specs,
  bpo::options_description vetos)
{
  auto proxy = options.add_options();
  for (auto const& spec : specs) {
    // skip everything found in the veto definition
    if (vetos.find_nothrow(spec.name, false) != nullptr) {
      continue;
    }

    switch (spec.type) {
      // FIXME: Should we handle int and size_t diffently?
      // FIXME: We should probably raise an error if the type is unknown
      case VariantType::Int:
        addConfigSpecOption<VariantType::Int>(spec, options);
        break;
      case VariantType::Int8:
        addConfigSpecOption<VariantType::Int8>(spec, options);
        break;
      case VariantType::Int16:
        addConfigSpecOption<VariantType::Int16>(spec, options);
        break;
      case VariantType::Int64:
        addConfigSpecOption<VariantType::Int64>(spec, options);
        break;
      case VariantType::UInt8:
        addConfigSpecOption<VariantType::UInt8>(spec, options);
        break;
      case VariantType::UInt16:
        addConfigSpecOption<VariantType::UInt16>(spec, options);
        break;
      case VariantType::UInt32:
        addConfigSpecOption<VariantType::UInt32>(spec, options);
        break;
      case VariantType::UInt64:
        addConfigSpecOption<VariantType::UInt64>(spec, options);
        break;
      case VariantType::Float:
        addConfigSpecOption<VariantType::Float>(spec, options);
        break;
      case VariantType::Double:
        addConfigSpecOption<VariantType::Double>(spec, options);
        break;
      case VariantType::String:
        addConfigSpecOption<VariantType::String>(spec, options);
        break;
      case VariantType::Bool:
        addConfigSpecOption<VariantType::Bool>(spec, options);
        break;
      case VariantType::ArrayInt:
        addConfigSpecOption<VariantType::ArrayInt>(spec, options);
        break;
      case VariantType::ArrayFloat:
        addConfigSpecOption<VariantType::ArrayFloat>(spec, options);
        break;
      case VariantType::ArrayDouble:
        addConfigSpecOption<VariantType::ArrayDouble>(spec, options);
        break;
      case VariantType::ArrayBool:
        addConfigSpecOption<VariantType::ArrayBool>(spec, options);
        break;
      case VariantType::ArrayString:
        addConfigSpecOption<VariantType::ArrayString>(spec, options);
        break;
      case VariantType::Array2DInt:
        addConfigSpecOption<VariantType::Array2DInt>(spec, options);
        break;
      case VariantType::Array2DFloat:
        addConfigSpecOption<VariantType::Array2DFloat>(spec, options);
        break;
      case VariantType::Array2DDouble:
        addConfigSpecOption<VariantType::Array2DDouble>(spec, options);
        break;
      case VariantType::LabeledArrayInt:
      case VariantType::LabeledArrayFloat:
      case VariantType::LabeledArrayDouble:
      // FIXME: for  Dict we should probably allow parsing stuff
      //        provided on the command line
      case VariantType::Dict:
      case VariantType::Unknown:
      case VariantType::Empty:
        break;
    };
  }
}

/// Check if option is defined
bool ConfigParamsHelper::hasOption(const std::vector<ConfigParamSpec>& specs, const std::string& optName)
{
  for (auto& old : specs) {
    if (old.name == optName) {
      return true;
    }
  }
  return false;
}

/// Add the ConfigParamSpec @a spec to @a specs if there is no parameter with
/// the same name already.
void ConfigParamsHelper::addOptionIfMissing(std::vector<ConfigParamSpec>& specs, const ConfigParamSpec& spec)
{
  if (!hasOption(specs, spec.name)) {
    specs.push_back(spec);
  }
}

/// populate boost program options making all options of type string
/// this is used for filtering the command line argument
bool ConfigParamsHelper::dpl2BoostOptions(const std::vector<ConfigParamSpec>& spec,
                                          boost::program_options::options_description& options,
                                          boost::program_options::options_description vetos)
{
  bool haveOption = false;
  for (const auto& configSpec : spec) {
    // skip everything found in the veto definition
    try {
      if (vetos.find_nothrow(configSpec.name, false) != nullptr) {
        continue;
      }
    } catch (boost::program_options::ambiguous_option& e) {
      for (auto const& alternative : e.alternatives()) {
        std::cerr << alternative << std::endl;
      }
      throw;
    }

    haveOption = true;
    std::stringstream defaultValue;
    defaultValue << configSpec.defaultValue;
    if (configSpec.type != VariantType::Bool) {
      if (configSpec.defaultValue.type() != VariantType::Empty) {
        options.add_options()(configSpec.name.c_str(),
                              bpo::value<std::string>()->default_value(defaultValue.str()),
                              configSpec.help.c_str());
      } else {
        options.add_options()(configSpec.name.c_str(),
                              bpo::value<std::string>(),
                              configSpec.help.c_str());
      }
    } else {
      if (configSpec.defaultValue.type() != VariantType::Empty) {
        options.add_options()(configSpec.name.c_str(),
                              bpo::value<bool>()->zero_tokens()->default_value(configSpec.defaultValue.get<bool>()),
                              configSpec.help.c_str());
      } else {
        options.add_options()(configSpec.name.c_str(),
                              bpo::value<bool>()->zero_tokens(),
                              configSpec.help.c_str());
      }
    }
  }

  return haveOption;
}

} // namespace o2::framework
