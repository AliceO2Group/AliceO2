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

namespace o2::event_visualisation
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

void VisualisationTrack::addPolyPoint(const float p[])
{
  mPolyX.push_back(p[0]);
  mPolyY.push_back(p[1]);
  mPolyZ.push_back(p[2]);
}

VisualisationCluster& VisualisationTrack::addCluster(float pos[])
{
  mClusters.emplace_back(pos, 0);
  return mClusters.back();
}

} // namespace o2
