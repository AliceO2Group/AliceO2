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

#include "Framework/ConfigContext.h"
#include <regex>
#include <sys/stat.h>
#include <unistd.h>

namespace o2::framework
{

ConfigContext::ConfigContext(ConfigParamRegistry& options, ServiceRegistryRef services, int argc, char** argv)
  : mOptions{options}, mServices{services}, mArgc{argc}, mArgv{argv} {}

bool ConfigContext::helpOnCommandLine() const
{
  bool helpasked = false;
  for (int argi = 0; argi < mArgc; ++argi) {
    if (strcmp(mArgv[argi], "--help") == 0 || (strcmp(mArgv[argi], "-h") == 0)) {
      helpasked = true;
      break;
    }
  }
  return helpasked;
}

bool ConfigContext::mayRun(std::string_view processor) const
{
  std::string_view id;
  for (int argi = 0; argi + 1 < mArgc; ++argi) {
    if (strcmp(mArgv[argi], "--id") == 0) {
      id = mArgv[argi + 1];
    }
  }
  // The driver, a device without a workflow on stdin and clones, which keep the
  // name of their template, build everything.
  struct stat s;
  if (id.empty() || fstat(STDIN_FILENO, &s) != 0 || !(S_ISFIFO(s.st_mode) || S_ISREG(s.st_mode)) || !mOptions.get<std::string>("clone").empty()) {
    return true;
  }
  // Pipelined devices get a _t<N> suffix, DDS ones a _dds<...> one.
  auto name = std::regex_replace(std::regex_replace(std::string{id}, std::regex{"_dds.*"}, ""), std::regex{"_t[0-9]+$"}, "");
  return id == processor || name == processor;
}

} // namespace o2::framework
