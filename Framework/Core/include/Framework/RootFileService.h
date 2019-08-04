// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_ROOTFILESERVICE_H
#define FRAMEWORK_ROOTFILESERVICE_H

#include "Framework/Variant.h"
#include <TFile.h>

#include <memory>
#include <map>
#include <string>
#include <vector>

namespace o2
{
namespace framework
{

/// A service which is delegated the creation and the booking of ROOT files.
/// A simple implementation is to use fmt as a local file name. More complex
/// implementations might have different sophisticated / centrally controlled
/// backends (e.g. have different filename for different process IDs allowing parallel
/// execution).
class RootFileService
{
 public:
  virtual std::shared_ptr<TFile> open(const char* fmt, ...) = 0;
  virtual std::string format(const char* fmt, ...) = 0;
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_ROOTFILESERVICE_H
