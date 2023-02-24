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
/// \file   Statistic.cxx
/// \brief  JSON statistic serialization
/// \author julian.myrcha@cern.ch

#include "EventVisualisationDataConverter/VisualisationEvent.h"
#include "EventVisualisationDataConverter/Statistic.h"
#include <fairlogger/Logger.h>
#include <iostream>
#include <iomanip>
#include <limits>

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
#include "EventVisualisationDataConverter/VisualisationEventJSONSerializer.h"
#include <fairlogger/Logger.h>

using namespace rapidjson;

namespace o2::event_visualisation
{

Statistic::Statistic() : tree(rapidjson::kObjectType)
{
  allocator = &tree.GetAllocator();
}
void Statistic::save(std::string fileName)
{
  // stringify
  rapidjson::StringBuffer buffer;
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
  tree.Accept(writer);
  std::string json = std::string(buffer.GetString(), buffer.GetSize());

  std::ofstream out(fileName);
  out << json;
  out.close();
}

void Statistic::toFile(const VisualisationEvent::Statistic& statistic)
{
  Value trackCount(kObjectType);
  for (int v = 0; v < o2::dataformats::GlobalTrackID::NSources; v++) {
    rapidjson::Value key(gDetectorSources[v].c_str(), gDetectorSources[v].length(), *allocator);
    trackCount.AddMember(key, rapidjson::Value().SetInt(statistic.mTrackCount[v]), *allocator);
  }
  tree.AddMember("trackCounts", trackCount, *allocator);
}
} // namespace o2::event_visualisation
