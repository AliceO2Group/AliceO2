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
#include "OptionsHelpers.h"
#include "Framework/Logger.h"
#include <boost/program_options.hpp>

namespace bpo = boost::program_options;

// Create a unique set of options and add them to the parser
// This is needed to avoid duplicate definitions in the main parser
// (e.g. if a device has a config option and a device-specific option)
// However, we complain if there are duplicate definitions which have
// different default values.
// Notice it's probably a good idea to simply specify string options
// and have the cast done in the user code. In any case what is
// passes on the command line is a string.
auto o2::framework::OptionsHelpers::makeUniqueOptions(bpo::options_description const& od) -> bpo::options_description
{
  bpo::options_description uniqueOptions;
  std::set<std::string> uniqueNames;
  std::map<std::string, std::string> optionDefaults;
  for (auto& option : od.options()) {
    if (uniqueNames.find(option->format_name()) == uniqueNames.end()) {
      uniqueOptions.add(option);
      uniqueNames.insert(option->format_name());
      boost::any defaultValue;
      option->semantic()->apply_default(defaultValue);
      // check if defaultValue is a string and if so, store it
      if (defaultValue.type() == typeid(std::string)) {
        optionDefaults.insert({option->format_name(), boost::any_cast<std::string>(defaultValue)});
      } else {
        optionDefaults.insert({option->format_name(), "not a string"});
      }
    } else {
      if (option->semantic()->max_tokens() == 1) {
        LOG(debug) << "Option " << option->format_name() << " is already defined, skipping";
        boost::any defaultValue1;
        option->semantic()->apply_default(defaultValue1);
        if (defaultValue1.type() != typeid(std::string)) {
          LOGP(error, "Option {} is already defined but it's not a string, please fix it. Actualy type {}", option->format_name(), defaultValue1.type().name());
        }
        auto defaultValueStr1 = boost::any_cast<std::string>(defaultValue1);
        auto defaultValueStr2 = optionDefaults.at(option->format_name());
        if (defaultValueStr2 == "not a string") {
          LOGP(error, "{} is duplicate but strings are the only supported duplicate values", option->format_name());
        }
        if (defaultValueStr1 != defaultValueStr2) {
          LOGP(error, "Option {} has different default values: {} and {}", option->format_name(), defaultValueStr1, defaultValueStr2);
        }
      }
    }
  }
  return uniqueOptions;
};
