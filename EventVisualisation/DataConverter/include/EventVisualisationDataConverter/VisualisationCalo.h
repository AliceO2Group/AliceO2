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

namespace o2::event_visualisation
{

class VisualisationCalo
{
  friend class VisualisationEventJSONSerializer;
  friend class VisualisationEventROOTSerializer;

 public:
  VisualisationCalo();

  /// constructor parametrisation (Value Object) for VisualisationCalo class
  struct VisualisationCaloVO {
    float time = 0;
    float energy = 0.0f;
    float phi = 0;
    float eta = 0;
    int PID = 0;
    o2::dataformats::GlobalTrackID gid = 0;
  };
  explicit VisualisationCalo(const VisualisationCaloVO& vo);

  VisualisationCalo(const VisualisationCalo& src);

  [[nodiscard]] float getEnergy() const { return mEnergy; }
  [[nodiscard]] float getTime() const { return mTime; }
  [[nodiscard]] int getPID() const { return mPID; }
  [[nodiscard]] o2::dataformats::GlobalTrackID getGID() const { return mBGID; }
  [[nodiscard]] o2::dataformats::GlobalTrackID::Source getSource() const { return static_cast<o2::dataformats::GlobalTrackID::Source>(mBGID.getSource()); }
  [[nodiscard]] float getPhi() const { return mPhi; }
  [[nodiscard]] float getEta() const { return mEta; }

 private:
  float mTime;   /// time
  float mEnergy; /// Energy of the particle
  int mPID;         /// PDG code of the particle
  float mEta; /// An angle from Z-axis to the radius vector pointing to the particle
  float mPhi; /// An angle from X-axis to the radius vector pointing to the particle
  o2::dataformats::GlobalTrackID mBGID;
};

} // namespace o2::event_visualisation

#endif // O2EVE_VISUALISATIONCALO_H
