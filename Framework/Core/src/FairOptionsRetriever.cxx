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
#include <options/FairMQProgOptions.h>
#include <boost/program_options.hpp>
#include <string>
#include <vector>

using namespace o2::framework;
namespace bpo = boost::program_options;

namespace o2
{
namespace framework
{

FairOptionsRetriever::FairOptionsRetriever(const FairMQProgOptions *opts)
: mOpts{opts}
{
}

int FairOptionsRetriever::getInt(const char *key) const {
  return mOpts->GetValue<int>(key);
}

float FairOptionsRetriever::getFloat(const char *key) const {
  return mOpts->GetValue<float>(key);
}

double FairOptionsRetriever::getDouble(const char *key) const {
  return mOpts->GetValue<double>(key);
}

bool FairOptionsRetriever::getBool(const char *key) const {
  return mOpts->GetValue<bool>(key);
}

std::string FairOptionsRetriever::getString(const char *key) const {
  return mOpts->GetValue<std::string>(key);
}

std::vector<std::string> FairOptionsRetriever::getVString(const char *key) const {
  assert(false && "Not implemented");
  return {};
}

} // namespace framework
} // namespace o2
