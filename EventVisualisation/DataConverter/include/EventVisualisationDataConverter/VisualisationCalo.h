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
/// \file    VisualisationCalo.h
/// \author  Julian Myrcha
///

#ifndef O2EVE_VISUALISATIONCALO_H
#define O2EVE_VISUALISATIONCALO_H

#include "rapidjson/document.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"

namespace o2
{
namespace event_visualisation
{
class VisualisationCalo
{
  friend class VisualisationEventJSONSerializer;
  friend class VisualisationEventROOTSerializer;

 public:
  // Default constructor
  VisualisationCalo();

  /// constructor parametrisation (Value Object) for VisualisationCalo class
  ///
  /// Simplifies passing parameters to constructor of VisualisationCalo
  /// by providing their names
  struct VisualisationCaloVO {
    float time = 0;
    float energy = 0.0f;
    float phi = 0;
    float eta = 0;
    int PID = 0;
    std::string gid = "";
    o2::dataformats::GlobalTrackID::Source source;
  };

  // Constructor with properties initialisation
  explicit VisualisationCalo(const VisualisationCaloVO& vo);

  VisualisationCalo(const VisualisationCalo& src);

  // Energy getter
  float getEnergy() const
  {
    return mEnergy;
  }

  // Time getter
  float getTime() const
  {
    return mTime;
  }

  // PID (particle identification code) getter
  int getPID() const
  {
    return mPID;
  }

  // GID  getter
  std::string getGIDAsString() const
  {
    return mGID;
  }

  // Source Getter
  o2::dataformats::GlobalTrackID::Source getSource() const
  {
    return mSource;
  }

  // Phi  getter
  float getPhi() const
  {
    return mPhi;
  }

  // Theta  getter
  float getEta() const
  {
    return mEta;
  }

 private:
  // Set coordinates of the beginning of the track
  void addStartCoordinates(const float xyz[3]);

  float mTime;   /// time
  float mEnergy; /// Energy of the particle

  int mPID;         /// PDG code of the particle
  std::string mGID; /// String representation of gid

  float mEta; /// An angle from Z-axis to the radius vector pointing to the particle
  float mPhi; /// An angle from X-axis to the radius vector pointing to the particle

  //  std::vector<int> mChildrenIDs; /// Unique IDs of children particles
  o2::dataformats::GlobalTrackID::Source mSource; /// data source of the track (debug)
};

} // namespace event_visualisation
} // namespace o2

#endif // O2EVE_VISUALISATIONCALO_H
