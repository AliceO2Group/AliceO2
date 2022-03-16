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
/// \file    VisualisationTrack.cxx
/// \author  Jeremi Niedziela
/// \author  Maciej Grochowicz
/// \author  Julian Myrcha
///

#include "EventVisualisationDataConverter/VisualisationTrack.h"
#include "FairLogger.h"

using namespace std;

namespace o2
{
namespace event_visualisation
{

VisualisationTrack::VisualisationTrack() = default;

VisualisationTrack::VisualisationTrack(const VisualisationTrackVO& vo)
{
  this->mCharge = vo.charge;
  this->mPID = vo.PID;
  this->mGID = vo.gid;
  this->mTheta = vo.theta;
  this->mPhi = vo.phi;
  this->mEta = vo.eta;
  this->addStartCoordinates(vo.startXYZ);
  this->mSource = vo.source;
  this->mTime = vo.time;
}

VisualisationTrack::VisualisationTrack(const VisualisationTrack& src)
{
  this->mCharge = src.mCharge;
  this->mPID = src.mPID;
  this->mGID = src.mGID;
  this->mTheta = src.mTheta;
  this->mPhi = src.mPhi;
  this->mEta = src.mEta;
  this->addStartCoordinates(src.getStartCoordinates());
  this->mSource = src.mSource;
  this->mTime = src.mTime;

  this->mPolyX = src.mPolyX;
  this->mPolyY = src.mPolyY;
  this->mPolyZ = src.mPolyZ;
  this->mClusters = src.mClusters;
}

void VisualisationTrack::addStartCoordinates(const float xyz[3])
{
  for (int i = 0; i < 3; i++) {
    mStartCoordinates[i] = xyz[i];
  }
}

void VisualisationTrack::addPolyPoint(float x, float y, float z)
{
  mPolyX.push_back(x);
  mPolyY.push_back(y);
  mPolyZ.push_back(z);
}

VisualisationTrack::VisualisationTrack(rapidjson::Value& tree)
{
  mClusters.clear();
  rapidjson::Value& jsonPolyX = tree["mPolyX"];
  rapidjson::Value& jsonPolyY = tree["mPolyY"];
  rapidjson::Value& jsonPolyZ = tree["mPolyZ"];
  rapidjson::Value& count = tree["count"];
  this->mCharge = 0;

  if (tree.HasMember("source")) {
    this->mSource = (o2::dataformats::GlobalTrackID::Source)tree["source"].GetInt();
  } else {
    this->mSource = o2::dataformats::GlobalTrackID::TPC; // temporary
  }
  this->mPID = tree["source"].GetInt();
  this->mTime = tree["time"].GetFloat();
  if (tree.HasMember("gid")) {
    this->mGID = tree["gid"].GetString();
  } else {
    this->mGID = "track";
  }
  this->mPolyX.reserve(count.GetInt());
  this->mPolyY.reserve(count.GetInt());
  this->mPolyZ.reserve(count.GetInt());
  for (auto& v : jsonPolyX.GetArray()) {
    mPolyX.push_back(v.GetDouble());
  }
  for (auto& v : jsonPolyY.GetArray()) {
    mPolyY.push_back(v.GetDouble());
  }
  for (auto& v : jsonPolyZ.GetArray()) {
    mPolyZ.push_back(v.GetDouble());
  }
  if (tree.HasMember("mClusters")) {
    rapidjson::Value& jsonClusters = tree["mClusters"];
    auto jsonArray = jsonClusters.GetArray();
    this->mClusters.reserve(jsonArray.Size());
    for (auto& v : jsonClusters.GetArray()) {
      mClusters.emplace_back(v);
    }
  }
}

rapidjson::Value VisualisationTrack::jsonTree(rapidjson::Document::AllocatorType& allocator)
{
  rapidjson::Value tree(rapidjson::kObjectType);
  rapidjson::Value jsonPolyX(rapidjson::kArrayType);
  rapidjson::Value jsonPolyY(rapidjson::kArrayType);
  rapidjson::Value jsonPolyZ(rapidjson::kArrayType);
  rapidjson::Value jsonStartCoordinates(rapidjson::kArrayType);

  tree.AddMember("count", rapidjson::Value().SetInt(this->getPointCount()), allocator);
  tree.AddMember("source", rapidjson::Value().SetInt(this->mSource), allocator);
  rapidjson::Value gid;
  gid.SetString(this->mGID.c_str(), this->mGID.size(), allocator);
  tree.AddMember("gid", gid, allocator);
  tree.AddMember("time", rapidjson::Value().SetFloat(isnan(this->mTime) ? 0 : this->mTime), allocator);
  tree.AddMember("charge", rapidjson::Value().SetInt(this->mCharge), allocator);
  tree.AddMember("theta", rapidjson::Value().SetFloat(isnan(this->mTheta) ? 0 : this->mTheta), allocator);
  tree.AddMember("phi", rapidjson::Value().SetFloat(isnan(this->mPhi) ? 0 : this->mPhi), allocator);
  tree.AddMember("eta", rapidjson::Value().SetFloat(isnan(this->mEta) ? 0 : this->mEta), allocator);
  tree.AddMember("PID", rapidjson::Value().SetInt(this->mPID), allocator);

  jsonStartCoordinates.PushBack((float)mStartCoordinates[0], allocator);
  jsonStartCoordinates.PushBack((float)mStartCoordinates[1], allocator);
  jsonStartCoordinates.PushBack((float)mStartCoordinates[2], allocator);
  tree.AddMember("jsonStartingXYZ", jsonStartCoordinates, allocator);

  for (size_t i = 0; i < this->getPointCount(); i++) {
    jsonPolyX.PushBack((float)mPolyX[i], allocator);
    jsonPolyY.PushBack((float)mPolyY[i], allocator);
    jsonPolyZ.PushBack((float)mPolyZ[i], allocator);
  }
  tree.AddMember("mPolyX", jsonPolyX, allocator);
  tree.AddMember("mPolyY", jsonPolyY, allocator);
  tree.AddMember("mPolyZ", jsonPolyZ, allocator);

  rapidjson::Value jsonClusters(rapidjson::kArrayType);
  for (size_t i = 0; i < this->mClusters.size(); i++) {
    jsonClusters.PushBack(this->mClusters[i].jsonTree(allocator), allocator);
  }
  tree.AddMember("mClusters", jsonClusters, allocator);

  return tree;
}

VisualisationCluster& VisualisationTrack::addCluster(float pos[])
{
  mClusters.emplace_back(pos, 0);
  return mClusters.back();
}

} // namespace event_visualisation
} // namespace o2
