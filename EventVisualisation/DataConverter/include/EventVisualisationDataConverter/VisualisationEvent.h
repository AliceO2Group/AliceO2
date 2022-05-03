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
/// \file    VisualisationEvent.h
/// \author  Jeremi Niedziela
/// \author  Maciej Grochowicz
/// \author  Julian Myrcha
///

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_VISUALISATIONEVENT_H
#define ALICE_O2_EVENTVISUALISATION_BASE_VISUALISATIONEVENT_H

#include "EventVisualisationDataConverter/VisualisationTrack.h"
#include "EventVisualisationDataConverter/VisualisationCluster.h"
#include "EventVisualisationDataConverter/VisualisationConstants.h"
#include <forward_list>
#include <ctime>

namespace o2
{
namespace event_visualisation
{

/// Minimalistic description of an event
///
/// This class is used mainly for visualisation purposes.
/// It stores simple information about tracks, V0s, kinks, cascades,
/// clusters and calorimeter towers, which can be used for visualisation
/// or exported for external applications.

class VisualisationEvent
{
 public:
  struct GIDVisualisation {
    bool contains[o2::dataformats::GlobalTrackID::NSources][o2::event_visualisation::EVisualisationGroup::NvisualisationGroups];
  };
  static GIDVisualisation mVis;
  std::string toJson();
  void fromJson(std::string json);
  bool fromFile(std::string fileName);
  VisualisationEvent();
  VisualisationEvent(std::string fileName);
  VisualisationEvent(const VisualisationEvent& source, EVisualisationGroup filter, float minTime, float maxTime);
  void toFile(std::string fileName);
  static std::string fileNameIndexed(const std::string fileName, const int index);

  /// constructor parametrisation (Value Object) for VisualisationEvent class
  ///
  /// Simplifies passing parameters to constructor of VisualisationEvent
  /// by providing their names
  struct VisualisationEventVO {
    int eventNumber;
    o2::header::DataHeader::RunNumberType runNumber;
    double energy;
    int multiplicity;
    std::string collidingSystem;
    time_t collisionTime;
  };
  // Default constructor
  VisualisationEvent(const VisualisationEventVO vo);

  VisualisationTrack* addTrack(VisualisationTrack::VisualisationTrackVO vo)
  {
    mTracks.emplace_back(vo);
    return &mTracks.back();
  }
  void remove_last_track() { mTracks.pop_back(); } // used to remove track assigned optimistically

  // Adds visualisation cluster inside visualisation event
  VisualisationCluster& addCluster(float XYZ[], float trackTime)
  {
    return mTracks.back().addCluster(XYZ);
  }

  VisualisationCluster& addCluster(float X, float Y, float Z, float trackTime)
  {
    float pos[] = {X, Y, Z};
    return mTracks.back().addCluster(pos);
  }

  // Multiplicity getter
  int GetMultiplicity() const
  {
    return mMultiplicity;
  }

  // Returns track with index i
  const VisualisationTrack& getTrack(int i) const
  {
    return mTracks[i];
  };

  // Returns number of tracks
  size_t getTrackCount() const
  {
    return mTracks.size();
  }

  // Clears event from stored data (tracks, collisions)
  void clear()
  {
    mTracks.clear();
    mClusters.clear();
  }

  const VisualisationCluster& getCluster(int i) const { return mClusters[i]; };
  size_t getClusterCount() const { return mClusters.size(); } // Returns number of clusters
  void setWorkflowVersion(float workflowVersion) { this->mWorkflowVersion = workflowVersion; }
  void setWorkflowParameters(const std::string& workflowParameters) { this->mWorkflowParameters = workflowParameters; }

  o2::header::DataHeader::RunNumberType getRunNumber() const { return this->mRunNumber; }
  void setRunNumber(o2::header::DataHeader::RunNumberType runNumber) { this->mRunNumber = runNumber; }

  o2::header::DataHeader::TForbitType getFirstTForbit() const { return this->mFirstTForbit; }
  void setFirstTForbit(o2::header::DataHeader::TForbitType firstTForbit) { this->mFirstTForbit = firstTForbit; }

  std::string getCollisionTime() const { return this->mCollisionTime; }
  void setCollisionTime(std::string collisionTime) { this->mCollisionTime = collisionTime; }

  float getMinTimeOfTracks() const { return this->mMinTimeOfTracks; }
  float getMaxTimeOfTracks() const { return this->mMaxTimeOfTracks; } /// maximum time of tracks in the event

  bool isEmpty() const { return getTrackCount() == 0 && getClusterCount() == 0; }

 private:
  float mMinTimeOfTracks;                            /// minimum time of tracks in the event
  float mMaxTimeOfTracks;                            /// maximum time of tracks in the event
  float mWorkflowVersion;                            /// workflow version used to generate this Event
  std::string mWorkflowParameters;                   /// workflow parameters used to generate this Event
  int mEventNumber;                                  /// event number in file
  o2::header::DataHeader::RunNumberType mRunNumber;  /// run number
  o2::header::DataHeader::TForbitType mFirstTForbit; /// first orbit in time frame
  double mEnergy;                                    /// energy of the collision
  int mMultiplicity;                                 /// number of particles reconstructed
  std::string mCollidingSystem;                      /// colliding system (e.g. proton-proton)
  std::string mCollisionTime;                        /// collision timestamp
  std::vector<VisualisationTrack> mTracks;           /// an array of visualisation tracks
  std::vector<VisualisationCluster> mClusters;       /// an array of visualisation clusters
};

} // namespace event_visualisation
} // namespace o2

#endif // ALICE_O2_EVENTVISUALISATION_BASE_VISUALISATIONEVENT_H
