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

#ifndef ALICEO2_TPC_LASERTRACK
#define ALICEO2_TPC_LASERTRACK

#include <string>
#include <gsl/span>

#include "CommonConstants/MathConstants.h"
#include "ReconstructionDataFormats/Track.h"

namespace o2::tpc
{
/// \class LaserTrack
/// This is the definition of the TPC Laser Track

class LaserTrack : public o2::track::TrackPar
{
 public:
  static constexpr int NumberOfTracks = 336;                                                   ///< Total number of laser tracks
  static constexpr int RodsPerSide = 6;                                                        ///< Number of laser rods per side
  static constexpr int BundlesPerRod = 4;                                                      ///< number of micro-mirror bundle per laser rod
  static constexpr int TracksPerBundle = 7;                                                    ///< number of micro-mirrors per bundle
  static constexpr float SectorSpanRad = o2::constants::math::SectorSpanRad;                   ///< secotor width in rad
  static constexpr std::array<float, 2> FirstRodPhi{2.f * SectorSpanRad, 1.f * SectorSpanRad}; ///< phi pos of first laser rod, A-, C-Side
  static constexpr float RodDistancePhi = 3.f * SectorSpanRad;                                 ///< phi distance of laser Rods
  static constexpr std::array<float, 4> CoarseBundleZPos{243.5, 165.5, 82, 11.5};              ///< coarse z-position of the laser bundles

  LaserTrack() = default;
  LaserTrack(int id, float x, float alpha, const std::array<float, o2::track::kNParams>& par) : mID(id), o2::track::TrackPar(x, alpha, par) { ; }
  LaserTrack(LaserTrack const&) = default;

  /// set laser track ID
  void setID(int id) { mID = id; }

  /// laser track ID
  int getID() const { return mID; }

  /// side of laser track
  int getSide() const { return 2 * getID() / NumberOfTracks; }

  /// laser rod number on Side
  int getRod() const { return (getID() / (BundlesPerRod * TracksPerBundle)) % RodsPerSide; }

  /// micro-mirror bundle inside laser rod
  int getBundle() const { return (getID() / TracksPerBundle) % BundlesPerRod; }

  /// laser beam inside mirror-mirror bundle
  int getBeam() const { return getID() % TracksPerBundle; }

 private:
  unsigned short mID{0}; ///< laser track ID

  ClassDefNV(LaserTrack, 1);
};

/// \class LaserTrackContainer
/// container class to hold all laser tracks
///
/// The definition of all tracks can be loaded from file using:
/// LaserTrackContainer c;
/// c.loadTracksFromFile();
class LaserTrackContainer
{
 public:
  LaserTrackContainer() = default;

  /// load laser tracks from file
  void loadTracksFromFile();

  /// get laser track
  /// \return LaserTrack
  LaserTrack const& getTrack(int id) const { return mLaserTracks[id]; }

  /// get container
  /// \return array of laser tracks
  const auto& getLaserTracks() const { return mLaserTracks; }

  /// dump tracks to a tree for simple visualization
  static void dumpToTree(const std::string_view fileName);

  /// get span with tracks in one bundle
  gsl::span<const LaserTrack> getTracksInBundle(int side, int rod, int bundle)
  {
    const int startID = LaserTrack::NumberOfTracks / 2 * side +
                        LaserTrack::BundlesPerRod * LaserTrack::TracksPerBundle * rod +
                        LaserTrack::TracksPerBundle * bundle;
    return gsl::span<const LaserTrack>(&mLaserTracks[startID], LaserTrack::TracksPerBundle);
  }

 private:
  std::array<LaserTrack, LaserTrack::NumberOfTracks> mLaserTracks;

  ClassDefNV(LaserTrackContainer, 1);
};

} // namespace o2::tpc
#endif
