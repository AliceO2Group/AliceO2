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
/// \file    VisualisationEventSerializer.h
/// \author  Julian Myrcha
///

#ifndef O2EVE_VISUALISATIONEVENTJSONSERIALIZER_H
#define O2EVE_VISUALISATIONEVENTJSONSERIALIZER_H

#include "EventVisualisationDataConverter/VisualisationEventSerializer.h"
#include "EventVisualisationDataConverter/VisualisationTrack.h"
#include <string>

namespace o2
{
namespace event_visualisation
{

class VisualisationEventJSONSerializer : public VisualisationEventSerializer
{
 public:
  static int getIntOrDefault(rapidjson::Value& tree, const char* key, int defaultValue = 0);
  static float getFloatOrDefault(rapidjson::Value& tree, const char* key, float defaultValue = 0.0f);
  static std::string getStringOrDefault(rapidjson::Value& tree, const char* key, const char* defaultValue = "");

 private:
  std::string toJson(const VisualisationEvent& event) const;
  void fromJson(VisualisationEvent& event, std::string json);

  // create calo from their JSON representation
  VisualisationCalo caloFromJSON(rapidjson::Value& tree);
  // create JSON representation of the calo
  rapidjson::Value jsonTree(const VisualisationCalo& calo, rapidjson::Document::AllocatorType& allocator) const;

  // create cluster from their JSON representation
  VisualisationCluster clusterFromJSON(rapidjson::Value& tree);
  rapidjson::Value jsonTree(const VisualisationCluster& cluster, rapidjson::Document::AllocatorType& allocator) const;

  // create track from their JSON representation
  VisualisationTrack trackFromJSON(rapidjson::Value& tree);
  // create JSON representation of the track
  rapidjson::Value jsonTree(const VisualisationTrack& track, rapidjson::Document::AllocatorType& allocator) const;

 public:
  bool fromFile(VisualisationEvent& event, std::string fileName) override;
  void toFile(const VisualisationEvent& event, std::string fileName) override;
  ~VisualisationEventJSONSerializer() override = default;
};

} // namespace event_visualisation
} // namespace o2

#endif // O2EVE_VISUALISATIONEVENTJSONSERIALIZER_H
