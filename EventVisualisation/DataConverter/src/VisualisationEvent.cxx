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
/// \author julian.myrcha@cern.ch
///

#include "EventVisualisationDataConverter/VisualisationEvent.h"

#include <string>
#include <limits>
#include <algorithm>

using namespace std;
using namespace rapidjson;

namespace o2::event_visualisation
{

VisualisationEvent::GIDVisualisation VisualisationEvent::mVis = [] {
  VisualisationEvent::GIDVisualisation res;
  for (auto filter = EVisualisationGroup::ITS;
       filter != EVisualisationGroup::NvisualisationGroups;
       filter = static_cast<EVisualisationGroup>(static_cast<int>(filter) + 1)) {
    if (filter == o2::event_visualisation::EVisualisationGroup::TPC) {
      res.contains[o2::dataformats::GlobalTrackID::TPC][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::ITSTPC][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::TPCTOF][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::TPCTRD][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::ITSTPCTRD][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::ITSTPCTOF][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::TPCTRDTOF][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::ITSTPCTRDTOF][filter] = true;
    }
    if (filter == o2::event_visualisation::EVisualisationGroup::ITS) {
      res.contains[o2::dataformats::GlobalTrackID::ITS][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::ITSTPC][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::ITSTPCTRD][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::ITSTPCTOF][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::ITSTPCTRDTOF][filter] = true;
    }
    if (filter == o2::event_visualisation::EVisualisationGroup::TRD) {
      res.contains[o2::dataformats::GlobalTrackID::TRD][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::TPCTRD][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::ITSTPCTRD][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::TPCTRDTOF][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::ITSTPCTRDTOF][filter] = true;
    }
    if (filter == o2::event_visualisation::EVisualisationGroup::TOF) {
      res.contains[o2::dataformats::GlobalTrackID::TOF][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::TPCTOF][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::ITSTPCTOF][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::TPCTRDTOF][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::ITSTPCTRDTOF][filter] = true;
    }
    if (filter == o2::event_visualisation::EVisualisationGroup::MFT) {
      res.contains[o2::dataformats::GlobalTrackID::MFT][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::MFTMCH][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::MFTMCHMID][filter] = true;
    }
    if (filter == o2::event_visualisation::EVisualisationGroup::MCH) {
      res.contains[o2::dataformats::GlobalTrackID::MCH][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::MFTMCH][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::MCHMID][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::MFTMCHMID][filter] = true;
    }
    if (filter == o2::event_visualisation::EVisualisationGroup::MID) {
      res.contains[o2::dataformats::GlobalTrackID::MID][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::MCHMID][filter] = true;
      res.contains[o2::dataformats::GlobalTrackID::MFTMCHMID][filter] = true;
    }
    if (filter == o2::event_visualisation::EVisualisationGroup::EMC) {
      res.contains[o2::dataformats::GlobalTrackID::EMC][filter] = true;
    }
    if (filter == o2::event_visualisation::EVisualisationGroup::PHS) {
      res.contains[o2::dataformats::GlobalTrackID::PHS][filter] = true;
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
  this->mCollisionTime = vo.collisionTime;
  this->mMinTimeOfTracks = numeric_limits<float>::max();
  this->mMaxTimeOfTracks = numeric_limits<float>::min();
}

void VisualisationEvent::appendAnotherEventCalo(const VisualisationEvent& another)
{
  for (auto calo : another.getCalorimetersSpan()) {
    this->mCalo.push_back(calo);
  }
}

VisualisationEvent::VisualisationEvent(const VisualisationEvent& source, EVisualisationGroup filter, float minTime, float maxTime)
{
  for (auto it = source.mTracks.begin(); it != source.mTracks.end(); ++it) {
    if (it->getTime() < minTime) {
      continue;
    }
    if (it->getTime() > maxTime) {
      continue;
    }
    if (VisualisationEvent::mVis.contains[it->getSource()][filter]) {
      this->mTracks.push_back(*it);
    }
  }
  for (auto it = source.mClusters.begin(); it != source.mClusters.end(); ++it) {
    if (VisualisationEvent::mVis.contains[it->getSource()][filter]) {
      this->mClusters.push_back(*it);
    }
  }
  for (auto it = source.mCalo.begin(); it != source.mCalo.end(); ++it) {
    if (VisualisationEvent::mVis.contains[it->getSource()][filter]) {
      this->mCalo.push_back(*it);
    }
  }
}

VisualisationEvent::VisualisationEvent()
{
  this->mCollisionTime = ""; // collision time not set
}

void VisualisationEvent::afterLoading()
{
  this->mMinTimeOfTracks = std::numeric_limits<float>::max();
  this->mMaxTimeOfTracks = std::numeric_limits<float>::min();
  for (auto& v : this->mTracks) {
    this->mMinTimeOfTracks = std::min(this->mMinTimeOfTracks, v.getTime());
    this->mMaxTimeOfTracks = std::max(this->mMaxTimeOfTracks, v.getTime());
  }
}

} // namespace o2
