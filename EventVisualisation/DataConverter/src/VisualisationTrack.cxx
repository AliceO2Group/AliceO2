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
  this->mTheta = vo.theta;
  this->mPhi = vo.phi;
  this->addStartCoordinates(vo.startXYZ);
  this->mSource = vo.source;
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
  rapidjson::Value& jsonPolyX = tree["mPolyX"];
  rapidjson::Value& jsonPolyY = tree["mPolyY"];
  rapidjson::Value& jsonPolyZ = tree["mPolyZ"];
  rapidjson::Value& count = tree["count"];
  this->mCharge = 0;

  this->mSource = (o2::dataformats::GlobalTrackID::Source)tree["source"].GetInt();
  this->mPID = (o2::dataformats::GlobalTrackID::Source)tree["source"].GetInt();
  //this->mTime = (o2::dataformats::GlobalTrackID::Source)tree["time"].GetFloat();
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
}

rapidjson::Value VisualisationTrack::jsonTree(rapidjson::Document::AllocatorType& allocator)
{
  rapidjson::Value tree(rapidjson::kObjectType);
  rapidjson::Value jsonPolyX(rapidjson::kArrayType);
  rapidjson::Value jsonPolyY(rapidjson::kArrayType);
  rapidjson::Value jsonPolyZ(rapidjson::kArrayType);

  tree.AddMember("count", rapidjson::Value().SetInt(this->getPointCount()), allocator);
  tree.AddMember("source", rapidjson::Value().SetInt(this->mSource), allocator);
  //tree.AddMember("time", rapidjson::Value().SetFloat(this->mTime), allocator);
  tree.AddMember("PID", rapidjson::Value().SetInt(this->mPID), allocator);

  for (size_t i = 0; i < this->getPointCount(); i++) {
    jsonPolyX.PushBack((float)mPolyX[i], allocator);
    jsonPolyY.PushBack((float)mPolyY[i], allocator);
    jsonPolyZ.PushBack((float)mPolyZ[i], allocator);
  }
  tree.AddMember("mPolyX", jsonPolyX, allocator);
  tree.AddMember("mPolyY", jsonPolyY, allocator);
  tree.AddMember("mPolyZ", jsonPolyZ, allocator);

  return tree;
}

} // namespace event_visualisation
} // namespace o2
