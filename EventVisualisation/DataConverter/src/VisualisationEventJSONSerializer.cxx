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
/// \file   VisualisationEventJSONSerializer.cxx
/// \brief  JSON serialization
/// \author julian.myrcha@cern.ch

#include "EventVisualisationDataConverter/VisualisationEventJSONSerializer.h"
#include <fairlogger/Logger.h>
#include <iostream>
#include <iomanip>
#include "ReconstructionDataFormats/GlobalTrackID.h"

#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"

using namespace rapidjson;

namespace o2::event_visualisation
{

void VisualisationEventJSONSerializer::toFile(const VisualisationEvent& event, std::string fileName)
{
  std::string json = toJson(event);
  std::ofstream out(fileName);
  out << json;
  out.close();
}

bool VisualisationEventJSONSerializer::fromFile(VisualisationEvent& event, std::string fileName)
{
  LOGF(info, "VisualisationEventJSONSerializer <- %s", fileName);
  if (FILE* file = fopen(fileName.c_str(), "r")) {
    fclose(file); // file exists
  } else {
    return false;
  }
  std::ifstream inFile;
  inFile.open(fileName);

  std::stringstream strStream;
  strStream << inFile.rdbuf(); // read the file
  inFile.close();
  std::string str = strStream.str(); // str holds the content of the file
  fromJson(event, str);
  return true;
}

std::string VisualisationEventJSONSerializer::toJson(const VisualisationEvent& event) const
{
  Document tree(kObjectType);
  Document::AllocatorType& allocator = tree.GetAllocator();

  // compatibility verification
  tree.AddMember("runNumber", rapidjson::Value().SetInt(event.mRunNumber), allocator);
  tree.AddMember("runType", rapidjson::Value().SetInt(event.mRunType), allocator);
  tree.AddMember("clMask", rapidjson::Value().SetInt(event.mClMask), allocator);
  tree.AddMember("trkMask", rapidjson::Value().SetInt(event.mTrkMask), allocator);
  tree.AddMember("tfCounter", rapidjson::Value().SetInt(event.mTfCounter), allocator);
  tree.AddMember("firstTForbit", rapidjson::Value().SetInt(event.mFirstTForbit), allocator);
  tree.AddMember("primaryVertex", rapidjson::Value().SetInt(event.mPrimaryVertex), allocator);
  std::string collisionTime = DateTime(event.mCreationTime);
  tree.AddMember("collisionTime", rapidjson::Value().SetString(collisionTime.c_str(), collisionTime.size()), allocator);
  tree.AddMember("creationTime", rapidjson::Value().SetUint64(event.mCreationTime), allocator);
  tree.AddMember("version", rapidjson::Value().SetUint64(event.mEveVersion), allocator);
  std::string version = std::to_string(event.mEveVersion / 100.0);
  version = version.substr(0, version.find('.') + 3);
  tree.AddMember("eveVersion", rapidjson::Value().SetString(version.c_str(), version.size()), allocator); // obsolete
  std::string workflowParameters = collisionTime + " t:" + bits(event.mTrkMask) + " c:" + bits(event.mClMask);
  tree.AddMember("workflowParameters", rapidjson::Value().SetString(workflowParameters.c_str(), workflowParameters.size()), allocator);
  // Tracks
  tree.AddMember("trackCount", rapidjson::Value().SetInt(event.getTrackCount()), allocator);

  Value jsonTracks(kArrayType);
  for (auto track : event.getTracksSpan()) {
    jsonTracks.PushBack(jsonTree(track, allocator), allocator);
  }
  tree.AddMember("mTracks", jsonTracks, allocator);

  // Clusters
  rapidjson::Value clusterCount(rapidjson::kNumberType);
  clusterCount.SetInt(event.getClusterCount());
  tree.AddMember("clusterCount", clusterCount, allocator);
  Value jsonClusters(kArrayType);
  for (auto cluster : event.getClustersSpan()) {
    jsonClusters.PushBack(jsonTree(cluster, allocator, nullptr), allocator);
  }
  tree.AddMember("mClusters", jsonClusters, allocator);

  // Calorimeters
  rapidjson::Value caloCount(rapidjson::kNumberType);
  caloCount.SetInt(event.getCaloCount());
  tree.AddMember("caloCount", caloCount, allocator);
  Value jsonCalos(kArrayType);
  for (auto calo : event.getCalorimetersSpan()) {
    jsonCalos.PushBack(jsonTree(calo, allocator), allocator);
  }
  tree.AddMember("mCalo", jsonCalos, allocator);

  // stringify
  rapidjson::StringBuffer buffer;
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
  tree.Accept(writer);
  std::string json_str = std::string(buffer.GetString(), buffer.GetSize());
  return json_str;
}

int VisualisationEventJSONSerializer::getIntOrDefault(rapidjson::Value& tree, const char* key, int defaultValue)
{
  if (tree.HasMember(key)) {
    rapidjson::Value& jsonValue = tree[key];
    return jsonValue.GetInt();
  }
  return defaultValue;
}
uint64_t VisualisationEventJSONSerializer::getUIntOrDefault(Value& tree, const char* key, uint64_t defaultValue)
{
  if (tree.HasMember(key)) {
    rapidjson::Value& jsonValue = tree[key];
    return jsonValue.GetUint64();
  }
  return defaultValue;
}

float VisualisationEventJSONSerializer::getFloatOrDefault(rapidjson::Value& tree, const char* key, float defaultValue)
{
  if (tree.HasMember(key)) {
    rapidjson::Value& jsonValue = tree[key];
    return jsonValue.GetFloat();
  }
  return defaultValue;
}

std::string VisualisationEventJSONSerializer::getStringOrDefault(rapidjson::Value& tree, const char* key, const char* defaultValue)
{
  if (tree.HasMember(key)) {
    rapidjson::Value& jsonValue = tree[key];
    return jsonValue.GetString();
  }
  return defaultValue;
}

void VisualisationEventJSONSerializer::fromJson(VisualisationEvent& event, std::string json)
{
  event.mTracks.clear();
  event.mClusters.clear();
  event.mCalo.clear();

  rapidjson::Document tree;
  tree.Parse(json.c_str());

  event.setRunNumber(getIntOrDefault(tree, "runNumber", 0));
  event.setRunType(static_cast<parameters::GRPECS::RunType>(getIntOrDefault(tree, "runType", 0)));
  event.setClMask(getIntOrDefault(tree, "clMask"));
  event.setTrkMask(getIntOrDefault(tree, "trkMask"));
  event.setTfCounter(getIntOrDefault(tree, "tfCounter"));
  event.setFirstTForbit(getIntOrDefault(tree, "firstTForbit"));
  event.setPrimaryVertex(getIntOrDefault(tree, "primaryVertex"));
  if (tree.HasMember("creationTime")) {
    event.setCreationTime(getUIntOrDefault(tree, "creationTime"));
  }
  if (event.mCreationTime == 0) {
    auto collisionTime = getStringOrDefault(tree, "collisionTime", "not specified");
    event.mCreationTime = parseDateTime(collisionTime.c_str());
  }
  event.mEveVersion = getIntOrDefault(tree, "version", 0);
  if (event.mEveVersion == 0) {
    std::string version = getStringOrDefault(tree, "eveVersion", "0.0");
    event.mEveVersion = (int)(100 * std::stof(version));
  }
  // event.setWorkflowParameters(getStringOrDefault(tree, "workflowParameters", "1.0"));

  rapidjson::Value& trackCount = tree["trackCount"];
  event.mTracks.reserve(trackCount.GetInt());
  rapidjson::Value& jsonTracks = tree["mTracks"];
  for (auto& v : jsonTracks.GetArray()) {
    event.mTracks.emplace_back(trackFromJSON(v));
  }

  if (tree.HasMember("caloCount")) {
    rapidjson::Value& caloCount = tree["caloCount"];
    event.mCalo.reserve(caloCount.GetInt());
    rapidjson::Value& jsonCalo = tree["mCalo"];
    for (auto& v : jsonCalo.GetArray()) {
      event.mCalo.emplace_back(caloFromJSON(v));
    }
  }

  rapidjson::Value& clusterCount = tree["clusterCount"];
  event.mClusters.reserve(clusterCount.GetInt());
  rapidjson::Value& jsonClusters = tree["mClusters"];
  for (auto& v : jsonClusters.GetArray()) {
    event.mClusters.emplace_back(clusterFromJSON(v, nullptr));
  }
  event.afterLoading();
}

VisualisationCluster VisualisationEventJSONSerializer::clusterFromJSON(rapidjson::Value& tree, const VisualisationTrack* track)
{
  float XYZ[3];
  float time;
  o2::dataformats::GlobalTrackID gid;
  rapidjson::Value& jsonX = tree["X"];
  rapidjson::Value& jsonY = tree["Y"];
  rapidjson::Value& jsonZ = tree["Z"];

  XYZ[0] = jsonX.GetDouble();
  XYZ[1] = jsonY.GetDouble();
  XYZ[2] = jsonZ.GetDouble();

  if (track) {
    time = track->mTime;
    gid = track->mBGID;
  } else {
    rapidjson::Value& jsonTime = tree["time"];
    rapidjson::Value& jsonBGID = tree["bgid"];
    time = jsonTime.GetDouble();
    gid = deserialize(jsonBGID.GetUint64());
  }

  VisualisationCluster cluster(XYZ, time, gid);
  return cluster;
}

rapidjson::Value VisualisationEventJSONSerializer::jsonTree(const VisualisationCluster& cluster, MemoryPoolAllocator<>& allocator, const VisualisationTrack* track) const
{
  rapidjson::Value tree(rapidjson::kObjectType);
  rapidjson::Value jsonX(rapidjson::kNumberType);
  rapidjson::Value jsonY(rapidjson::kNumberType);
  rapidjson::Value jsonZ(rapidjson::kNumberType);
  jsonX.SetDouble(cluster.mCoordinates[0]);
  jsonY.SetDouble(cluster.mCoordinates[1]);
  jsonZ.SetDouble(cluster.mCoordinates[2]);
  tree.AddMember("X", jsonX, allocator);
  tree.AddMember("Y", jsonY, allocator);
  tree.AddMember("Z", jsonZ, allocator);
  if (track == nullptr) {
    unsigned bgid = serialize(cluster.mBGID);
    tree.AddMember("bgid", rapidjson::Value().SetUint(bgid), allocator); // proper (binary) gid storing
    tree.AddMember("time", rapidjson::Value().SetFloat(std::isnan(cluster.mTime) ? 0 : cluster.mTime), allocator);
  }
  return tree;
}

VisualisationCalo VisualisationEventJSONSerializer::caloFromJSON(rapidjson::Value& tree)
{
  VisualisationCalo calo;
  calo.mTime = tree["time"].GetFloat();
  calo.mEnergy = tree["energy"].GetFloat();
  calo.mEta = tree["eta"].GetFloat();
  calo.mPhi = tree["phi"].GetFloat();
  if (tree.HasMember("bgid")) {
    unsigned bgid = tree["bgid"].GetUint();
    calo.mBGID = deserialize(bgid);
  } else {
    auto source = tree["source"].GetUint();
    calo.mBGID = deserialize(source, 0, 0);
  }
  calo.mPID = tree["PID"].GetInt();
  return calo;
}

rapidjson::Value VisualisationEventJSONSerializer::jsonTree(const VisualisationCalo& calo, rapidjson::MemoryPoolAllocator<>& allocator) const
{
  rapidjson::Value tree(rapidjson::kObjectType);
  // tree.AddMember("source", rapidjson::Value().SetInt(calo.mSource), allocator);
  tree.AddMember("time", rapidjson::Value().SetFloat(std::isnan(calo.mTime) ? 0 : calo.mTime), allocator);
  tree.AddMember("energy", rapidjson::Value().SetFloat(calo.mEnergy), allocator);
  tree.AddMember("eta", rapidjson::Value().SetFloat(std::isnan(calo.mEta) ? 0 : calo.mEta), allocator);
  tree.AddMember("phi", rapidjson::Value().SetFloat(std::isnan(calo.mPhi) ? 0 : calo.mPhi), allocator);
  unsigned bgid = serialize(calo.mBGID);
  tree.AddMember("bgid", rapidjson::Value().SetUint(bgid), allocator);
  tree.AddMember("source", rapidjson::Value().SetUint(calo.mBGID.getSource()), allocator);
  tree.AddMember("PID", rapidjson::Value().SetInt(calo.mPID), allocator);
  return tree;
}

VisualisationTrack VisualisationEventJSONSerializer::trackFromJSON(rapidjson::Value& tree)
{
  VisualisationTrack track;
  track.mClusters.clear();
  rapidjson::Value& jsonStartingXYZ = tree["jsonStartingXYZ"];
  rapidjson::Value& jsonPolyX = tree["mPolyX"];
  rapidjson::Value& jsonPolyY = tree["mPolyY"];
  rapidjson::Value& jsonPolyZ = tree["mPolyZ"];
  rapidjson::Value& count = tree["count"];
  track.mCharge = getIntOrDefault(tree, "charge", 0);
  track.mTheta = getFloatOrDefault(tree, "theta", 0);
  track.mPhi = getFloatOrDefault(tree, "phi", 0);
  track.mEta = getFloatOrDefault(tree, "eta", 0);
  // track.mSource = (o2::dataformats::GlobalTrackID::Source)getIntOrDefault(tree, "source", (int)o2::dataformats::GlobalTrackID::TPC);
  track.mPID = getIntOrDefault(tree, "PID", 0);
  track.mTime = tree["time"].GetFloat();
  track.mBGID = 0;
  if (tree.HasMember("bgid")) {
    track.mBGID = deserialize(getUIntOrDefault(tree, "bgid", 0));
  } else {
    std::string gid = getStringOrDefault(tree, "gid", "track");
    if (gid != "track") {
      track.mBGID = gidFromString(gid); // json uses text gid format only
    }
  }
  track.mPolyX.reserve(count.GetInt());
  track.mPolyY.reserve(count.GetInt());
  track.mPolyZ.reserve(count.GetInt());
  auto startingXYZ = jsonStartingXYZ.GetArray();
  track.mStartCoordinates[0] = startingXYZ[0].GetFloat();
  track.mStartCoordinates[1] = startingXYZ[1].GetFloat();
  track.mStartCoordinates[2] = startingXYZ[2].GetFloat();
  for (auto& v : jsonPolyX.GetArray()) {
    track.mPolyX.push_back(v.GetDouble());
  }
  for (auto& v : jsonPolyY.GetArray()) {
    track.mPolyY.push_back(v.GetDouble());
  }
  for (auto& v : jsonPolyZ.GetArray()) {
    track.mPolyZ.push_back(v.GetDouble());
  }
  if (tree.HasMember("mClusters")) {
    rapidjson::Value& jsonClusters = tree["mClusters"];
    auto jsonArray = jsonClusters.GetArray();
    track.mClusters.reserve(jsonArray.Size());
    for (auto& v : jsonClusters.GetArray()) {
      track.mClusters.emplace_back(clusterFromJSON(v, &track));
    }
  }
  return track;
}

rapidjson::Value VisualisationEventJSONSerializer::jsonTree(const VisualisationTrack& track, rapidjson::Document::AllocatorType& allocator) const
{
  rapidjson::Value tree(rapidjson::kObjectType);
  rapidjson::Value jsonPolyX(rapidjson::kArrayType);
  rapidjson::Value jsonPolyY(rapidjson::kArrayType);
  rapidjson::Value jsonPolyZ(rapidjson::kArrayType);
  rapidjson::Value jsonStartCoordinates(rapidjson::kArrayType);

  tree.AddMember("count", rapidjson::Value().SetInt(track.getPointCount()), allocator);

  rapidjson::Value gid;
  std::string stringGID = track.mBGID.asString();
  gid.SetString(stringGID.c_str(), stringGID.size(), allocator); // obsolete - for compatibility
  tree.AddMember("gid", gid, allocator);
  unsigned bgid = serialize(track.mBGID);
  tree.AddMember("bgid", rapidjson::Value().SetUint(bgid), allocator); // proper (binary) gid storing

  tree.AddMember("time", rapidjson::Value().SetFloat(std::isnan(track.mTime) ? 0 : track.mTime), allocator);
  tree.AddMember("charge", rapidjson::Value().SetInt(track.mCharge), allocator);
  tree.AddMember("theta", rapidjson::Value().SetFloat(std::isnan(track.mTheta) ? 0 : track.mTheta), allocator);
  tree.AddMember("phi", rapidjson::Value().SetFloat(std::isnan(track.mPhi) ? 0 : track.mPhi), allocator);
  tree.AddMember("eta", rapidjson::Value().SetFloat(std::isnan(track.mEta) ? 0 : track.mEta), allocator);
  tree.AddMember("PID", rapidjson::Value().SetInt(track.mPID), allocator);

  jsonStartCoordinates.PushBack((float)track.mStartCoordinates[0], allocator);
  jsonStartCoordinates.PushBack((float)track.mStartCoordinates[1], allocator);
  jsonStartCoordinates.PushBack((float)track.mStartCoordinates[2], allocator);
  tree.AddMember("jsonStartingXYZ", jsonStartCoordinates, allocator);

  for (size_t i = 0; i < track.getPointCount(); i++) {
    jsonPolyX.PushBack((float)track.mPolyX[i], allocator);
    jsonPolyY.PushBack((float)track.mPolyY[i], allocator);
    jsonPolyZ.PushBack((float)track.mPolyZ[i], allocator);
  }
  tree.AddMember("mPolyX", jsonPolyX, allocator);
  tree.AddMember("mPolyY", jsonPolyY, allocator);
  tree.AddMember("mPolyZ", jsonPolyZ, allocator);

  rapidjson::Value jsonClusters(rapidjson::kArrayType);

  for (auto cluster : track.getClustersSpan()) {
    jsonClusters.PushBack(jsonTree(cluster, allocator, &track), allocator);
  }
  tree.AddMember("mClusters", jsonClusters, allocator);

  return tree;
}

} // namespace o2::event_visualisation
