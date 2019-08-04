// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/LocalRootFileService.h"
#include "Framework/Logger.h"

#include <cstdio>
#include <cstdarg>
#include <string>
#include <climits>
#include <cassert>

namespace o2
{
namespace framework
{

// All we do is to printout
std::shared_ptr<TFile> LocalRootFileService::open(const char* format, ...)
{
  char buffer[PATH_MAX];
  va_list arglist;
  va_start(arglist, format);
  vsnprintf(buffer, PATH_MAX, format, arglist);
  va_end(arglist);
  return std::make_shared<TFile>(buffer, "recreate");
}

std::string LocalRootFileService::format(const char* format, ...)
{
  char buffer[PATH_MAX];
  va_list arglist;
  va_start(arglist, format);
  vsnprintf(buffer, PATH_MAX, format, arglist);
  va_end(arglist);
  return std::move(std::string(buffer));
}

} // namespace framework
} // namespace o2
