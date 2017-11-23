// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

namespace o2 {
namespace framework {

void populateBoostProgramOptions(
    boost::program_options::options_description &options,
    const std::vector<ConfigParamSpec> &specs
  );

/// populate boost program options making all options of type string
/// this is used for filtering the command line argument
bool
prepareOptionsDescription(const std::vector<ConfigParamSpec> &spec,
                          boost::program_options::options_description& options);

/// populate boost program options for a complete workflow
template<typename ContainerType>
boost::program_options::options_description
prepareOptionDescriptions(const ContainerType &workflow)
{
  boost::program_options::options_description specOptions("Spec groups");
  for (const auto & spec : workflow) {
    std::string help = "Usage: --" + spec.name + R"( "spec options")";
    specOptions.add_options()(spec.name.c_str(),
                              boost::program_options::value<std::string>(),
                              help.c_str());
    boost::program_options::options_description options(spec.name.c_str());
    if (prepareOptionsDescription(spec.options, options)) {
      specOptions.add(options);
    }
  }
  return specOptions;
}

}
}
#endif // FRAMEWORK_CONFIGPARAMSHELPER_H
