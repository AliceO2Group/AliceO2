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

/// \file FileWatcher.h
/// \brief Observing folder for created and removed files - preserving current
/// \author julian.myrcha@cern.ch

#ifndef O2EVE_DIRECTORYLOADER_H
#define O2EVE_DIRECTORYLOADER_H

#include <string>
#include <deque>

namespace o2
{
namespace event_visualisation
{

class DirectoryLoader
{
 public:
  static std::deque<std::string> load(const std::string& path, const std::string& marker);
};

} // namespace event_visualisation
} // namespace o2

#endif // O2EVE_DIRECTORYLOADER_H
