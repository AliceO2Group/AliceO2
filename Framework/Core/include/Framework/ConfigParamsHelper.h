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
#ifndef FRAMEWORK_CONFIGPARAMSHELPER_H
#define FRAMEWORK_CONFIGPARAMSHELPER_H

#include "Framework/ConfigParamSpec.h"
#include <boost/program_options.hpp>

#include <string>
#include <vector>
#include <string>
#include <type_traits>

namespace o2::framework
{

using options_description = boost::program_options::options_description;

struct ConfigParamsHelper {
  static void populateBoostProgramOptions(
    options_description& options,
    const std::vector<ConfigParamSpec>& specs,
    options_description vetos = options_description());

  /// populate boost program options making all options of type string
  /// this is used for filtering the command line argument
  /// all options which are found in the vetos are skipped
  static bool dpl2BoostOptions(const std::vector<ConfigParamSpec>& spec,
                               options_description& options,
                               boost::program_options::options_description vetos = options_description());

  /// Add the ConfigParamSpec @a spec to @a specs if there is no parameter with
  /// the same name already.
  static void addOptionIfMissing(std::vector<ConfigParamSpec>& specs, ConfigParamSpec spec);

  /// populate boost program options for a complete workflow
  template <typename ContainerType>
  static boost::program_options::options_description
    prepareOptionDescriptions(ContainerType const& workflow,
                              std::vector<ConfigParamSpec> const& currentWorkflowOptions,
                              options_description vetos = options_description(),
                              std::string mode = "full")
  {
    boost::program_options::options_description toplevel;
    boost::program_options::options_description wo("Global workflow options");
    if (dpl2BoostOptions(currentWorkflowOptions, wo, vetos)) {
      toplevel.add(wo);
    }
    std::string specOptionsDescription = "Available data processors";
    if (mode == "short") {
      specOptionsDescription += " (full info with '--help full')";
    }
    options_description specOptions(specOptionsDescription);
    for (const auto& spec : workflow) {
      std::string name = "Data processor options: " + spec.name;
      boost::program_options::options_description processorOptions(name);
      if (dpl2BoostOptions(spec.options, processorOptions, vetos)) {
        // if vetos have been provided to the function we also need to make
        // sure that there are no duplicate option definitions for the individual
        // processor specs, so we add in order to be vetos for all subsequent specs.
        // Note: this only concerns the main parser, all individual options are
        // handled when starting individual processors.
        if (vetos.options().size() > 0) {
          vetos.add(processorOptions);
        }
        if (mode == "full") {
          specOptions.add(processorOptions);
        } else if (mode == spec.name) {
          toplevel.add(processorOptions);
          break;
        }
      }
      if (mode == "full" || mode == "short") {
        std::string help;
        if (mode == "full") {
          help = "Option groups by process name: --" + spec.name + R"( "<processor options>")";
        } else if (mode == "short" && processorOptions.options().size() > 0) {
          help = "Use '--help " + spec.name + "' to display processor options";
        } else if (mode == "short" && processorOptions.options().size() == 0) {
          help = "No processor options";
        }
        specOptions.add_options()(spec.name.c_str(),
                                  boost::program_options::value<std::string>(),
                                  help.c_str());
      }
    }
    if (workflow.size() > 0 && (mode == "full" || mode == "short")) {
      toplevel.add(specOptions);
    }
    return toplevel;
  }

  template <VariantType V>
  static void addConfigSpecOption(const ConfigParamSpec& spec,
                                  boost::program_options::options_description& options)
  {
    const char* name = spec.name.c_str();
    const char* help = spec.help.c_str();

    if constexpr (V == VariantType::Int ||
                  V == VariantType::Int64 ||
                  V == VariantType::Float ||
                  V == VariantType::Double ||
                  V == VariantType::Bool) {
      using Type = typename variant_type<V>::type;
      using BoostType = typename std::conditional<V == VariantType::String, std::string, Type>::type;
      auto value = boost::program_options::value<BoostType>();
      value = value->default_value(spec.defaultValue.get<BoostType>());
      if constexpr (V == VariantType::Bool) {
        // for bool values we also support the zero_token option to make
        // the option usable as a single switch
        value = value->zero_tokens();
      }
      options.add_options()(name, value, help);
    } else if constexpr (V == VariantType::ArrayInt ||
                         V == VariantType::ArrayFloat ||
                         V == VariantType::ArrayDouble ||
                         V == VariantType::ArrayBool ||
                         V == VariantType::ArrayString ||
                         V == VariantType::Array2DInt ||
                         V == VariantType::Array2DFloat ||
                         V == VariantType::Array2DDouble) {
      auto value = boost::program_options::value<std::string>();
      value = value->default_value(spec.defaultValue.asString());
      if constexpr (V != VariantType::String) {
        value = value->multitoken();
      }
      options.add_options()(name, value, help);
    } else {
      using Type = typename variant_type<V>::type;
      using BoostType = typename std::conditional<V == VariantType::String, std::string, Type>::type;
      auto value = boost::program_options::value<BoostType>();
      options.add_options()(name, value, help);
    }
  }
};

} // namespace o2::framework
#endif // FRAMEWORK_CONFIGPARAMSHELPER_H
