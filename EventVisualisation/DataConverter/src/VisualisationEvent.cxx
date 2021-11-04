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
/// \file    VisualisationEvent.cxx
/// \author  Jeremi Niedziela
/// \author  Maciej Grochowicz
///

#include "EventVisualisationDataConverter/VisualisationEvent.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"

#include <string>
#include <iostream>
#include <iomanip>
#include "FairLogger.h"

using namespace std;
using namespace rapidjson;

namespace o2
{
namespace event_visualisation
{
constexpr int JSON_FILE_VERSION = 1;

VisualisationEvent::GIDVisualisation VisualisationEvent::mVis = [] {
  VisualisationEvent::GIDVisualisation res;
  for (auto filter = EVisualisationGroup::ITS;
       filter != EVisualisationGroup::NvisualisationGroups;
       filter = static_cast<EVisualisationGroup>(static_cast<int>(filter) + 1)) {
    if (filter == o2::event_visualisation::EVisualisationGroup::TPC) {
      res.contains[o2::dataformats::GlobalTrackID::TPC][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::ITSTPC][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::ITSTPCTRD][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::ITSTPCTRDTOF][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::TPCTRD][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::TPCTRDTOF][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::TPCTOF][filter] = true;
    }
    if (filter == o2::event_visualisation::EVisualisationGroup::ITS) {
      res.contains[o2::dataformats::GlobalTrackID::ITS][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::ITSTPC][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::ITSTPCTRD][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::ITSTPCTRDTOF][filter] = true;
    }
    if (filter == o2::event_visualisation::EVisualisationGroup::TRD) {
      res.contains[o2::dataformats::GlobalTrackID::TRD][filter] = true;
    }
    if (filter == o2::event_visualisation::EVisualisationGroup::TOF) {
      res.contains[o2::dataformats::GlobalTrackID::ITSTPCTRDTOF][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::TPCTRDTOF][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::TPCTOF][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::TOF][filter] = true;
    }
    if (filter == o2::event_visualisation::EVisualisationGroup::MFT) {
      res.contains[o2::dataformats::GlobalTrackID::MFT][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::MFTMCH][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::MFTMCHMID][filter] = true;
    }
    if (filter == o2::event_visualisation::EVisualisationGroup::MCH) {
      res.contains[o2::dataformats::GlobalTrackID::MCH][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::MFTMCH][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::MFTMCHMID][filter] = true;
    }
    if (filter == o2::event_visualisation::EVisualisationGroup::MID) {
      res.contains[o2::dataformats::GlobalTrackID::MCH][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::MFTMCH][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::MFTMCHMID][filter] = true;
    }
  }
  return res;
}();

/// Ctor -- set the minimalistic event up
VisualisationEvent::VisualisationEvent(VisualisationEventVO vo)
{
  this->mEventNumber = vo.eventNumber;
  this->mRunNumber = vo.runNumber;
  this->mEnergy = vo.energy;
  this->mMultiplicity = vo.multiplicity;
  this->mCollidingSystem = vo.collidingSystem;
  this->mTimeStamp = vo.timeStamp;
}

std::string VisualisationEvent::toJson()
{
  Document tree(kObjectType);
  Document::AllocatorType& allocator = tree.GetAllocator();

  // compatibility verification
  tree.AddMember("fileVersion", rapidjson::Value().SetInt(JSON_FILE_VERSION), allocator);
  //tree.AddMember("timeStamp", rapidjson::Value().SetFloat(this->mTimeStamp), allocator);
  tree.AddMember("workflowVersion", rapidjson::Value().SetFloat(this->mWorkflowVersion), allocator);
  tree.AddMember("workflowParameters", rapidjson::Value().SetString(this->mWorkflowParameters.c_str(), this->mWorkflowParameters.size()), allocator);
  // Tracks
  tree.AddMember("trackCount", rapidjson::Value().SetInt(this->getTrackCount()), allocator);

  Value jsonTracks(kArrayType);
  for (size_t i = 0; i < this->getTrackCount(); i++) {
    jsonTracks.PushBack(this->mTracks[i].jsonTree(allocator), allocator);
  }
  tree.AddMember("mTracks", jsonTracks, allocator);

  // Clusters
  rapidjson::Value clusterCount(rapidjson::kNumberType);
  clusterCount.SetInt(this->getClusterCount());
  tree.AddMember("clusterCount", clusterCount, allocator);
  Value jsonClusters(kArrayType);
  for (size_t i = 0; i < this->getClusterCount(); i++) {
    jsonClusters.PushBack(this->mClusters[i].jsonTree(allocator), allocator);
  }
  tree.AddMember("mClusters", jsonClusters, allocator);

  // stringify
  rapidjson::StringBuffer buffer;
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
  tree.Accept(writer);
  std::string json_str = std::string(buffer.GetString(), buffer.GetSize());
  return json_str;
}

VisualisationEvent::VisualisationEvent(std::string fileName)
{
  this->fromFile(fileName);
}

VisualisationEvent::VisualisationEvent(const VisualisationEvent& source, EVisualisationGroup filter)
{
  for (auto it = source.mTracks.begin(); it != source.mTracks.end(); ++it) {
    if (VisualisationEvent::mVis.contains[it->getSource()][filter]) {
      this->addTrack({.time = it->getTime(),
                      .charge = it->getCharge(),
                      .PID = it->getPID(),
                      .startXYZ = {
                        it->getStartCoordinates()[0], it->getStartCoordinates()[1], it->getStartCoordinates()[2]},
                      .phi = it->getPhi(),
                      .theta = it->getTheta(),
                      .source = it->getSource()});
    }
  }
  for (auto it = source.mClusters.begin(); it != source.mClusters.end(); ++it) {
  }
}

void VisualisationEvent::fromJson(std::string json)
{
  mTracks.clear();
  mClusters.clear();

  rapidjson::Document tree;
  tree.Parse(json.c_str());

  auto version = 1;
  if (tree.HasMember("fileVersion")) {
    rapidjson::Value& fileVersion = tree["fileVersion"];
    version = fileVersion.GetInt();
  }
  auto timeStamp = time(nullptr);
  if (tree.HasMember("timeStamp")) {
    rapidjson::Value& fileTimeStamp = tree["timeStamp"];
    timeStamp = fileTimeStamp.GetFloat();
  }
  this->mTimeStamp = timeStamp;

  rapidjson::Value& trackCount = tree["trackCount"];
  this->mTracks.reserve(trackCount.GetInt());
  rapidjson::Value& jsonTracks = tree["mTracks"];
  for (auto& v : jsonTracks.GetArray()) {
    mTracks.emplace_back(v);
  }

  rapidjson::Value& clusterCount = tree["clusterCount"];
  this->mClusters.reserve(clusterCount.GetInt());
  rapidjson::Value& jsonClusters = tree["mClusters"];
  for (auto& v : jsonClusters.GetArray()) {
    mClusters.emplace_back(v);
  }
}

void VisualisationEvent::toFile(std::string fileName)
{
  std::string json = toJson();
  std::ofstream out(fileName);
  out << json;
  out.close();
}

std::string VisualisationEvent::fileNameIndexed(const std::string fileName, const int index)
{
  std::stringstream buffer;
  buffer << fileName << std::setfill('0') << std::setw(3) << index << ".json";
  return buffer.str();
}

bool VisualisationEvent::fromFile(std::string fileName)
{
  if (FILE* file = fopen(fileName.c_str(), "r")) {
    fclose(file); // file exists
  } else {
    return false;
  }
  std::ifstream inFile;
  inFile.open(fileName);

  std::stringstream strStream;
  strStream << inFile.rdbuf(); //read the file
  inFile.close();
  std::string str = strStream.str(); //str holds the content of the file
  fromJson(str);
  return true;
}

VisualisationEvent::VisualisationEvent()
{
  this->mTimeStamp = time(nullptr); // current time
}

} // namespace event_visualisation
} // namespace o2
