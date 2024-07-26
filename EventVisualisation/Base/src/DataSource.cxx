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

/// \file DataSource.cxx
/// \brief reading from file(s) base class
/// \author j.myrcha@cern.ch

#include <EventVisualisationBase/DataSource.h>

#include <fairlogger/Logger.h>
// #include <time.h>

namespace o2::event_visualisation
{

std::string DataSource::getCreationTimeAsString() const
{
  char buffer[90];
  time_t time = this->mCreationTime;
  const char* format = "%a %b %-d %H:%M:%S %Y";
  struct tm* timeinfo = localtime(&time);
  strftime(buffer, sizeof(buffer), format, timeinfo);
  return buffer;
}

} // namespace o2::event_visualisation
