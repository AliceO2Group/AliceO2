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

namespace o2 {
namespace framework {

void populateBoostProgramOptions(
    boost::program_options::options_description &options,
    const std::vector<ConfigParamSpec> &specs
  );
}

}
#endif // FRAMEWORK_CONFIGPARAMSHELPER_H
