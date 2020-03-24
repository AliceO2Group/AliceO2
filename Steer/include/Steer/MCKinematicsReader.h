// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "SimulationDataFormat/DigitizationContext.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <vector>

class TChain;

namespace o2
{
namespace steer
{

class MCKinematicsReader
{
 public:
  /// default constructor
  MCKinematicsReader() = default;

  /// constructor taking context and auto-initializing
  MCKinematicsReader(std::string_view filename)
  {
    init(filename);
  }

  /// inits the reader from a digitization context
  /// returns true if successful
  bool init(std::string_view filename);

  bool isInitialized() const { return mInitialized; }

  /// query an MC track given a basic label object
  /// returns nullptr if no track was found
  MCTrack const* getTrack(o2::MCCompLabel const&) const;

  /// query an MC track given a basic label object
  /// returns nullptr if no track was found
  MCTrack const* getTrack(int source, int event, int track) const;

  /// variant returning all tracks for source and event at once
  std::vector<MCTrack> const& getTracks(int source, int event) const;

  /// get all primaries for a certain event

  /// get all secondaries of the given label

  /// get all mothers/daughters of the given label

 private:
  void loadTracksForSource(int source) const;

  DigitizationContext const* mDigitizationContext = nullptr;

  // chains for each source
  std::vector<TChain*> mInputChains;

  // a vector of tracks foreach source and each collision
  mutable std::vector<std::vector<std::vector<o2::MCTrack>>> mTracks; // the in-memory track container

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
  if (mTracks[source].size() == 0) {
    loadTracksForSource(source);
  }
  return &mTracks[source][event][track];
}

inline std::vector<MCTrack> const& MCKinematicsReader::getTracks(int source, int event) const
{
  if (mTracks[source].size() == 0) {
    loadTracksForSource(source);
  }
  return mTracks[source][event];
}

} // namespace steer
} // namespace o2
