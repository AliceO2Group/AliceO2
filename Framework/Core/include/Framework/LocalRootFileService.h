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
#ifndef FRAMEWORK_LOCALROOTFILESERVICE_H
#define FRAMEWORK_LOCALROOTFILESERVICE_H

#include "Framework/RootFileService.h"
#include "TFile.h"

#include <map>
#include <string>
#include <vector>
#include <memory>

namespace o2
{
namespace framework
{

/// A simple service to create ROOT files in the local folder
class LocalRootFileService : public RootFileService
{
 public:
  std::shared_ptr<TFile> open(const char* fmt, ...) final;
  std::string format(const char* fmt, ...) final;
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_LOCALROOTFILESERVICE_H
