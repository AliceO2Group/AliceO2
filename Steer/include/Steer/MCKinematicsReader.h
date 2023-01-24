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

#ifndef MC_KINEMATICS_READER_H
#define MC_KINEMATICS_READER_H

#include "SimulationDataFormat/DigitizationContext.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/TrackReference.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include <vector>

class TChain;

namespace o2
{

namespace steer
{

class MCKinematicsReader
{
 public:
  enum class Mode {
    kDigiContext,
    kMCKine
  };

  /// default constructor
  MCKinematicsReader() = default;

  /// destructor
  ~MCKinematicsReader();

  /// constructor taking a name and mode (either kDigiContext or kMCKine)
  /// In case of "context", the name is the filename of the digitization context.
  /// In case of MCKine mode, the name is the "prefix" referencing a single simulation production.
  /// The default mode is kDigiContext.
  MCKinematicsReader(std::string_view name, Mode mode = Mode::kDigiContext)
  {
    if (mode == Mode::kMCKine) {
      initFromKinematics(name);
    } else if (mode == Mode::kDigiContext) {
      initFromDigitContext(name);
    }
  }

  /// inits the reader from a digitization context
  /// returns true if successful
  bool initFromDigitContext(std::string_view filename);

  /// inits the reader from a simple kinematics file
  bool initFromKinematics(std::string_view filename);

  bool isInitialized() const { return mInitialized; }

  /// query an MC track given a basic label object
  /// returns nullptr if no track was found
  MCTrack const* getTrack(o2::MCCompLabel const&) const;

  /// query an MC track given source, event, track IDs
  /// returns nullptr if no track was found
  MCTrack const* getTrack(int source, int event, int track) const;

  /// query an MC track given event, track IDs
  /// returns nullptr if no track was found
  MCTrack const* getTrack(int event, int track) const;

  /// variant returning all tracks for source and event at once
  std::vector<MCTrack> const& getTracks(int source, int event) const;

  /// API to ask releasing tracks (freeing memory) for source + event
  void releaseTracksForSourceAndEvent(int source, int event);

  /// variant returning all tracks for an event id (source = 0) at once
  std::vector<MCTrack> const& getTracks(int event) const;

  /// get all primaries for a certain event

  /// get all secondaries of the given label

  /// get all mothers/daughters of the given label

  /// return all track references associated to a source/event/track
  gsl::span<o2::TrackReference> getTrackRefs(int source, int event, int track) const;
  /// return all track references associated to a source/event
  const std::vector<o2::TrackReference>& getTrackRefsByEvent(int source, int event) const;
  /// return all track references associated to a event/track (when initialized from kinematics directly)
  gsl::span<o2::TrackReference> getTrackRefs(int event, int track) const;

  /// retrieves the MCEventHeader for a given eventID and sourceID
  o2::dataformats::MCEventHeader const& getMCEventHeader(int source, int event) const;

  /// Get number of sources
  size_t getNSources() const;

  /// Get number of events
  size_t getNEvents(int source) const;

  DigitizationContext const* getDigitizationContext() const
  {
    return mDigitizationContext;
  }

 private:
  void initTracksForSource(int source) const;
  void loadTracksForSourceAndEvent(int source, int eventID) const;
  void loadHeadersForSource(int source) const;
  void loadTrackRefsForSource(int source) const;
  void initIndexedTrackRefs(std::vector<o2::TrackReference>& refs, o2::dataformats::MCTruthContainer<o2::TrackReference>& indexedrefs) const;

  DigitizationContext const* mDigitizationContext = nullptr;

  // chains for each source
  std::vector<TChain*> mInputChains;

  // a vector of tracks foreach source and each collision
  mutable std::vector<std::vector<std::vector<o2::MCTrack>*>> mTracks;                                       // the in-memory track container
  mutable std::vector<std::vector<o2::dataformats::MCEventHeader>> mHeaders;                                 // the in-memory header container
  mutable std::vector<std::vector<o2::dataformats::MCTruthContainer<o2::TrackReference>>> mIndexedTrackRefs; // the in-memory track ref container

  bool mInitialized = false; // whether initialized
};

inline MCTrack const* MCKinematicsReader::getTrack(o2::MCCompLabel const& label) const
{
  const auto source = label.getSourceID();
  const auto event = label.getEventID();
  const auto track = label.getTrackID();
  return getTrack(source, event, track);
}

inline MCTrack const* MCKinematicsReader::getTrack(int source, int event, int track) const
{
  return &getTracks(source, event)[track];
}

inline MCTrack const* MCKinematicsReader::getTrack(int event, int track) const
{
  return getTrack(0, event, track);
}

inline std::vector<MCTrack> const& MCKinematicsReader::getTracks(int source, int event) const
{
  if (mTracks[source].size() == 0) {
    initTracksForSource(source);
  }
  if (mTracks[source][event] == nullptr) {
    loadTracksForSourceAndEvent(source, event);
  }
  return *mTracks[source][event];
}

inline std::vector<MCTrack> const& MCKinematicsReader::getTracks(int event) const
{
  return getTracks(0, event);
}

inline o2::dataformats::MCEventHeader const& MCKinematicsReader::getMCEventHeader(int source, int event) const
{
  if (mHeaders.at(source).size() == 0) {
    loadHeadersForSource(source);
  }
  return mHeaders.at(source)[event];
}

inline gsl::span<o2::TrackReference> MCKinematicsReader::getTrackRefs(int source, int event, int track) const
{
  if (mIndexedTrackRefs[source].size() == 0) {
    loadTrackRefsForSource(source);
  }
  return mIndexedTrackRefs[source][event].getLabels(track);
}

inline const std::vector<o2::TrackReference>& MCKinematicsReader::getTrackRefsByEvent(int source, int event) const
{
  if (mIndexedTrackRefs[source].size() == 0) {
    loadTrackRefsForSource(source);
  }
  return mIndexedTrackRefs[source][event].getTruthArray();
}

inline gsl::span<o2::TrackReference> MCKinematicsReader::getTrackRefs(int event, int track) const
{
  return getTrackRefs(0, event, track);
}

inline size_t MCKinematicsReader::getNSources() const
{
  return mTracks.size();
}

inline size_t MCKinematicsReader::getNEvents(int source) const
{
  if (mTracks[source].size() == 0) {
    initTracksForSource(source);
  }
  return mTracks[source].size();
}

} // namespace steer
} // namespace o2

#endif
