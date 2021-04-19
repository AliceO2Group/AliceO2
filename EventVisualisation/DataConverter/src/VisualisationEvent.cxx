// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "rapidjson/stringbuffer.h"
#include <fstream>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace rapidjson;

namespace o2
{
namespace event_visualisation
{

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

  // Tracks
  rapidjson::Value trackCount(rapidjson::kNumberType);
  trackCount.SetInt(this->getTrackCount());
  tree.AddMember("trackCount", trackCount, allocator);
  Value jsonTracks(kArrayType);
  for (int i = 0; i < this->getTrackCount(); i++)
    jsonTracks.PushBack(this->mTracks[i].jsonTree(allocator), allocator);
  tree.AddMember("mTracks", jsonTracks, allocator);

  // Clusters
  rapidjson::Value clusterCount(rapidjson::kNumberType);
  clusterCount.SetInt(this->getClusterCount());
  tree.AddMember("clusterCount", clusterCount, allocator);
  Value jsonClusters(kArrayType);
  for (int i = 0; i < this->getClusterCount(); i++)
    jsonClusters.PushBack(this->mClusters[i].jsonTree(allocator), allocator);
  tree.AddMember("mClusters", jsonClusters, allocator);

  // stringify
  StringBuffer buffer;
  Writer<StringBuffer> writer(buffer);
  tree.Accept(writer);
  return buffer.GetString();
}

VisualisationEvent::VisualisationEvent(std::string fileName)
{
  this->fromFile(fileName);
}

void VisualisationEvent::fromJson(std::string json)
{
  mTracks.clear();
  mClusters.clear();

  rapidjson::Document tree;
  tree.Parse(json.c_str());

  rapidjson::Value& trackCount = tree["trackCount"];
  this->mTracks.reserve(trackCount.GetInt());
  rapidjson::Value& jsonTracks = tree["mTracks"];
  for (auto& v : jsonTracks.GetArray())
    mTracks.emplace_back(v);

  rapidjson::Value& clusterCount = tree["clusterCount"];
  this->mClusters.reserve(clusterCount.GetInt());
  rapidjson::Value& jsonClusters = tree["mClusters"];
  for (auto& v : jsonClusters.GetArray())
    mClusters.emplace_back(v);
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
  if (FILE* file = fopen(fileName.c_str(), "r"))
    fclose(file); // file exists
  else
    return false;
  std::ifstream inFile;
  inFile.open(fileName);

  std::stringstream strStream;
  strStream << inFile.rdbuf(); //read the file
  inFile.close();
  std::string str = strStream.str(); //str holds the content of the file
  fromJson(str);
  return true;
}

} // namespace event_visualisation
} // namespace o2
