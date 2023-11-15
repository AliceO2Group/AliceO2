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

/// \file DataSourceOffline.cxx
/// \brief Grouping reading from file(s)
/// \author p.nowakowski@cern.ch

#include <EventVisualisationBase/DataSourceOffline.h>

#include <TSystem.h>
#include <TEveTreeTools.h>
#include <TEveTrack.h>
#include <TEveManager.h>
#include <TFile.h>
#include <TPRegexp.h>
#include <TObject.h>
#include <fmt/core.h>
#include <fairlogger/Logger.h>

namespace o2
{
namespace event_visualisation
{
DataSourceOffline::DataSourceOffline(std::string const aodCoverterPath, std::string const path, std::string const file, bool hideGui) : DataSourceOnline({path})
{
  std::string const batch = hideGui ? "-b" : "";
  std::string command = fmt::format("{} {} --aod-file {} --jsons-folder \"{}\"", aodCoverterPath, batch, file, path);

  auto retVal = std::system(command.c_str());
  (void)retVal;
}

} // namespace event_visualisation
} // namespace o2
