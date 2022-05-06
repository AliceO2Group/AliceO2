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
/// \file    VisualisationTrack.h
/// \author  Jeremi Niedziela
/// \author  Maciej Grochowicz
/// \author  Julian Myrcha
///

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_VISUALISATIONTRACK_H
#define ALICE_O2_EVENTVISUALISATION_BASE_VISUALISATIONTRACK_H

#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ConversionConstants.h"
#include "VisualisationCluster.h"
#include "rapidjson/document.h"

#include <iosfwd>
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <gsl/span>

namespace o2
{
namespace event_visualisation
{

/// Minimalistic description of particles track
///
/// This class is used mainly for visualisation purpose.
/// It keeps basic information about a track, such as its vertex,
/// momentum, PID, phi and theta or helix curvature.

class VisualisationTrack
{
  friend class VisualisationEventJSONSerializer;
  friend class VisualisationEventROOTSerializer;

 public:
  // Default constructor
  VisualisationTrack();

  /// constructor parametrisation (Value Object) for VisualisationTrack class
  ///
  /// Simplifies passing parameters to constructor of VisualisationTrack
  /// by providing their names
  struct VisualisationTrackVO {
    float time = 0;
    int charge = 0;
    int PID = 0;
    float startXYZ[3];
    float phi = 0;
    float theta = 0;
    float eta = 0;
    std::string gid = "";
    o2::dataformats::GlobalTrackID::Source source;
  };
  // Constructor with properties initialisation
  VisualisationTrack(const VisualisationTrackVO& vo);

  VisualisationTrack(const VisualisationTrack& src);

  // Add child particle (coming from decay of this particle)
  void addChild(int childID);
  // Add xyz coordinates of the point along the track
  void addPolyPoint(float x, float y, float z);
  // Time getter
  float getTime() const { return mTime; }
  // Charge getter
  int getCharge() const { return mCharge; }
  // PID (particle identification code) getter
  int getPID() const { return mPID; }
  // GID  getter
  std::string getGIDAsString() const { return mGID; }
  // Source Getter
  o2::dataformats::GlobalTrackID::Source getSource() const { return mSource; }
  // Phi  getter
  float getPhi() const { return mPhi; }
  // Theta  getter
  float getTheta() const { return mTheta; }
  //
  const float* getStartCoordinates() const { return mStartCoordinates; }

  size_t getPointCount() const { return mPolyX.size(); }
  std::array<float, 3> getPoint(size_t i) const { return std::array<float, 3>{mPolyX[i], mPolyY[i], mPolyZ[i]}; }

  VisualisationCluster& addCluster(float pos[]);
  const VisualisationCluster& getCluster(int i) const { return mClusters[i]; };
  size_t getClusterCount() const { return mClusters.size(); } // Returns number of clusters
  gsl::span<const VisualisationCluster> getClustersSpan() const
  {
    return mClusters;
  }

 private:
  // Set coordinates of the beginning of the track
  void addStartCoordinates(const float xyz[3]);

  float mTime;                 /// track time
  int mCharge;                 /// Charge of the particle

  int mPID;                    /// PDG code of the particle
  std::string mGID;            /// String representation of gid

  float mStartCoordinates[3]; /// Vector of track's start coordinates

  float mTheta; /// An angle from Z-axis to the radius vector pointing to the particle
  float mPhi;   /// An angle from X-axis to the radius vector pointing to the particle
  float mEta;

  //  std::vector<int> mChildrenIDs; /// Uniqe IDs of children particles
  o2::dataformats::GlobalTrackID::Source mSource; /// data source of the track (debug)

  /// Polylines -- array of points along the trajectory of the track
  std::vector<float> mPolyX;
  std::vector<float> mPolyY;
  std::vector<float> mPolyZ;

  std::vector<VisualisationCluster> mClusters; /// an array of visualisation clusters belonging to track
};

} // namespace event_visualisation
} // namespace o2
#endif // ALICE_O2_EVENTVISUALISATION_BASE_VISUALISATIONTRACK_H
