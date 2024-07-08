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
#include "EventVisualisationDataConverter/VisualisationCalo.h"
#include "EventVisualisationDataConverter/VisualisationConstants.h"
#include "DataFormatsParameters/ECSDataAdapters.h"
#include "Framework/DataProcessingHeader.h"
#include <forward_list>
#include <ctime>
#include <gsl/span>

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
  friend class VisualisationEventJSONSerializer;
  friend class VisualisationEventROOTSerializer;
  friend class VisualisationEventOpenGLSerializer;

 public:
  struct GIDVisualisation {
    bool contains[o2::dataformats::GlobalTrackID::NSources][o2::event_visualisation::EVisualisationGroup::NvisualisationGroups];
  };
  static GIDVisualisation mVis;
  VisualisationEvent();
  VisualisationEvent(const VisualisationEvent& source, EVisualisationGroup filter, float minTime, float maxTime);

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
  explicit VisualisationEvent(const VisualisationEventVO vo);

  void appendAnotherEventCalo(const VisualisationEvent& another);

  VisualisationTrack* addTrack(VisualisationTrack::VisualisationTrackVO vo)
  {
    mTracks.emplace_back(vo);
    return &mTracks.back();
  }

  VisualisationCluster& addCluster(const float *xyz)
  {
    //float pos[] = {X, Y, Z};
    return mTracks.back().addCluster(xyz);
  }

  void addGlobalCluster(float* xyz, float time, o2::dataformats::GlobalTrackID gid)
  {
    mClusters.emplace_back(xyz, time, gid);
  }

  void addGlobalCluster(const TVector3& xyz, o2::dataformats::GlobalTrackID gid, float time)
  {
    auto result = mClusters.emplace_back(xyz);
    result.mBGID = gid;
    result.mTime = time;
  }

  VisualisationCalo* addCalo(VisualisationCalo::VisualisationCaloVO vo)
  {
    mCalo.emplace_back(vo);
    return &mCalo.back();
  }

  // Multiplicity getter
  int GetMultiplicity() const
  {
    return mMultiplicity;
  }

  // Returns track with index i
  [[nodiscard]] const VisualisationTrack& getTrack(int i) const
  {
    return mTracks[i];
  };

  // Returns number of tracks
  size_t getTrackCount() const
  {
    return mTracks.size();
  }

  gsl::span<const VisualisationCluster> getClustersSpan() const
  {
    return mClusters;
  }

  gsl::span<const VisualisationTrack> getTracksSpan() const
  {
    return mTracks;
  }

  gsl::span<const VisualisationCalo> getCalorimetersSpan() const
  {
    return mCalo;
  }

  size_t getCaloCount() const
  {
    return mCalo.size();
  }

  // Returns number of tracks with detector contribution (including standalone)
  size_t getDetectorTrackCount(o2::detectors::DetID::ID id) const
  {
    return getDetectorsTrackCount(o2::detectors::DetID::getMask(id));
  }

  // Returns number of tracks with any detector contribution (including standalone) from the list
  size_t getDetectorsTrackCount(o2::detectors::DetID::mask_t mdet) const
  {
    return std::count_if(mTracks.begin(), mTracks.end(), [&](const auto& t) {
      return (o2::dataformats::GlobalTrackID::getSourceDetectorsMask(t.getSource()) & mdet).any();
    });
  }

  // Returns number of tracks from a given source
  size_t getSourceTrackCount(o2::dataformats::GlobalTrackID::Source src) const
  {
    return std::count_if(mTracks.begin(), mTracks.end(), [&](const auto& t) {
      return t.getSource() == src;
    });
  }

  // Clears event from stored data (tracks, collisions)
  void clear()
  {
    mTracks.clear();
    mClusters.clear();
    mCalo.clear();
  }

  void afterLoading(); // compute internal fields which are not persisted

  const VisualisationCluster& getCluster(int i) const { return mClusters[i]; };
  size_t getClusterCount() const { return mClusters.size(); } // Returns number of clusters
  // void setWorkflowParameters(const std::string& workflowParameters) { this->mWorkflowParameters = workflowParameters; }

  // std::string getCollisionTime() const { return DateTime(this->mCreationTime); }
  // void setCollisionTime(std::string collisionTime) { this->mCreationTime = this->parseDateTime(collisionTime.c_str()); }

  o2::framework::DataProcessingHeader::CreationTime getCreationTime() const { return this->mCreationTime; }
  void setCreationTime(o2::framework::DataProcessingHeader::CreationTime creationTime) { this->mCreationTime = creationTime; }

  void setEveVersion(int eveVersion) { this->mEveVersion = eveVersion; }

  float getMinTimeOfTracks() const { return this->mMinTimeOfTracks; }
  float getMaxTimeOfTracks() const { return this->mMaxTimeOfTracks; } /// maximum time of tracks in the event

  bool isEmpty() const { return getTrackCount() == 0 && getClusterCount() == 0; }

  int getClMask() const { return mClMask; }
  void setClMask(int value) { mClMask = value; }

  int getTrkMask() const { return mTrkMask; }
  void setTrkMask(int value) { mTrkMask = value; }

  o2::header::DataHeader::RunNumberType getRunNumber() const { return this->mRunNumber; }
  void setRunNumber(o2::header::DataHeader::RunNumberType runNumber) { this->mRunNumber = runNumber; }

  o2::parameters::GRPECS::RunType getRunType() const { return this->mRunType; }
  void setRunType(o2::parameters::GRPECS::RunType runType) { this->mRunType = runType; }

  o2::header::DataHeader::TFCounterType getTfCounter() const { return this->mTfCounter; }
  void setTfCounter(o2::header::DataHeader::TFCounterType value) { this->mTfCounter = value; }

  [[nodiscard]] o2::header::DataHeader::TForbitType getFirstTForbit() const { return this->mFirstTForbit; }
  void setFirstTForbit(o2::header::DataHeader::TForbitType value) { this->mFirstTForbit = value; }

  std::size_t getPrimaryVertex() const { return this->mPrimaryVertex; }
  void setPrimaryVertex(std::size_t pv) { this->mPrimaryVertex = pv; }

  VisualisationEvent limit(std::size_t maximum_number_of_items);

 private:
  o2::framework::DataProcessingHeader::CreationTime mCreationTime; /// creation time in binary format
  int mClMask;                                      /// clusters requested during aquisition
  int mTrkMask;                                     /// tracks requested during aquisition
  o2::header::DataHeader::RunNumberType mRunNumber; /// run number
  o2::header::DataHeader::TFCounterType mTfCounter;
  o2::header::DataHeader::TForbitType mFirstTForbit;
  o2::parameters::GRPECS::RunType mRunType;
  std::size_t mPrimaryVertex;

  float mMinTimeOfTracks;                           /// minimum time of tracks in the event
  float mMaxTimeOfTracks;                           /// maximum time of tracks in the event
  int mEveVersion;                                  /// workflow version used to generate this Event (120 -> 1.20)
  // std::string mWorkflowParameters;                  /// workflow parameters used to generate this Event
  int mEventNumber;                                 /// event number in file
  double mEnergy;                                   /// energy of the collision
  int mMultiplicity;                                /// number of particles reconstructed
  std::string mCollidingSystem;                     /// colliding system (e.g. proton-proton)
  // std::string mCollisionTime;                       /// collision timestamp
  std::vector<VisualisationTrack> mTracks;          /// an array of visualisation tracks
  std::vector<VisualisationCluster> mClusters;      /// an array of visualisation clusters
  std::vector<VisualisationCalo> mCalo;             /// an array of visualisation calorimeters
};

} // namespace event_visualisation
} // namespace o2

#endif // ALICE_O2_EVENTVISUALISATION_BASE_VISUALISATIONEVENT_H
