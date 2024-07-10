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

///
/// \file   VisualisationEventSerializer.cxx
/// \brief  Serialization VisualisationEvent
/// \author julian.myrcha@cern.ch

#include "EventVisualisationDataConverter/VisualisationEventSerializer.h"
#include "EventVisualisationDataConverter/VisualisationEventJSONSerializer.h"
#include "EventVisualisationDataConverter/VisualisationEventROOTSerializer.h"
#include "EventVisualisationDataConverter/VisualisationEventOpenGLSerializer.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <map>

namespace o2::event_visualisation
{
std::map<std::string, VisualisationEventSerializer*> VisualisationEventSerializer::instances = {
  {".json", new o2::event_visualisation::VisualisationEventJSONSerializer()},
  {".root", new o2::event_visualisation::VisualisationEventROOTSerializer()},
  {".eve", new o2::event_visualisation::VisualisationEventOpenGLSerializer()}};

std::string VisualisationEventSerializer::fileNameIndexed(const std::string fileName, const int index)
{
  std::stringstream buffer;
  buffer << fileName << std::setfill('0') << std::setw(3) << index << ".json";
  return buffer.str();
}

o2::dataformats::GlobalTrackID VisualisationEventSerializer::deserialize(unsigned int serializedValue)
{
  o2::dataformats::GlobalTrackID result;
  *((unsigned*)&(result)) = serializedValue;
  return result;
}

unsigned VisualisationEventSerializer::serialize(o2::dataformats::GlobalTrackID gidValue)
{
  unsigned result;
  result = *((unsigned*)&(gidValue));
  return result;
}

o2::dataformats::GlobalTrackID
  VisualisationEventSerializer::deserialize(unsigned int source, unsigned int index, unsigned int flags)
{
  return serialize(index + source * (1 << 25) + flags * (1 << 30));
}

o2::dataformats::GlobalTrackID VisualisationEventSerializer::gidFromString(const std::string& gid)
{
  static std::map<std::string, int> sources = {
    {"ITS", 0},
    {"TPC", 1},
    {"TRD", 2},
    {"TOF", 3},
    {"PHS", 4},
    {"CPV", 5},
    {"EMC", 6},
    {"HMP", 7},
    {"MFT", 8},
    {"MCH", 9},
    {"MID", 10},
    {"ZDC", 11},
    {"FT0", 12},
    {"FV0", 13},
    {"FDD", 14},
    {"ITS-TPC", 15},
    {"TPC-TOF", 16},
    {"TPC-TRD", 17},
    {"MFT-MCH", 18},
    {"ITS-TPC-TRD", 19},
    {"ITS-TPC-TOF", 20},
    {"TPC-TRD-TOF", 21},
    {"MFT-MCH-MID", 22},
    {"ITS-TPC-TRD-TOF", 23}, // full barrel track
    {"ITSAB", 24},           // ITS AfterBurner tracklets
    {"CTP", 25}};
  const auto first = gid.find('/');
  const auto second = gid.find('/', first + 1);
  auto source = sources[gid.substr(1, first - 1)];
  auto index = std::stoi(gid.substr(first + 1, second - 1));
  auto flags = std::stoi(gid.substr(second + 1, gid.size() - 1));
  return index + source * (1 << 25) + flags * (1 << 30);
}

time_t VisualisationEventSerializer::parseDateTime(const char* datetimeString)
{
  std::string date(datetimeString);
  date += " GMT";
  const char* format = "%A %B %d %H:%M:%S %Y %Z";
  struct tm tmStruct;
  strptime(datetimeString, format, &tmStruct);
  return mktime(&tmStruct);
}

std::string VisualisationEventSerializer::DateTime(time_t time)
{
  char buffer[90];
  const char* format = "%a %b %-d %H:%M:%S %Y";
  struct tm* timeinfo = localtime(&time);
  strftime(buffer, sizeof(buffer), format, timeinfo);
  return buffer;
}

std::string VisualisationEventSerializer::bits(unsigned number)
{
  char result[33];
  result[32] = 0;
  unsigned mask = 1;
  for (int i = 31; i >= 0; i--) {
    result[i] = (mask & number) ? '1' : '0';
    mask <<= 1;
  }
  return result;
}

} // namespace o2::event_visualisation
