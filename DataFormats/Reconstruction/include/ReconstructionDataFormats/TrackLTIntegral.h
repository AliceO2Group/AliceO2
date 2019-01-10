// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackLTIntegral.h
/// \brief Track Length and TOF integral
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_TRACK_LTINTEGRAL_H_
#define ALICEO2_TRACK_LTINTEGRAL_H_

#include <Rtypes.h>
#include "ReconstructionDataFormats/PID.h"
#include "ReconstructionDataFormats/Track.h"

namespace o2
{
namespace track
{

class TrackPar;

class TrackLTIntegral
{
 public:
  TrackLTIntegral() = default;
  TrackLTIntegral(const TrackLTIntegral& stc) = default;
  ~TrackLTIntegral() = default;

  static constexpr int getNTOFs() { return o2::track::PID::NIDs; }

  float getL() const { return mL; }
  float getTOF(int id) const { return mT[id]; }

  void clear()
  {
    mL = 0;
    for (int i = getNTOFs(); i--;) {
      mT[i] = 0.f;
    }
  }

  void addStep(float dL, const TrackPar& track);

  void setL(float l) { mL = l; }
  void setTOF(float t, int id) { mT[id] = t; }

  void print() const;

 private:
  float mL = 0.;                           // length in cm
  float mT[o2::track::PID::NIDs] = { 0. }; // TOF in ps

  ClassDefNV(TrackLTIntegral, 1);
};
}; // namespace track
}; // namespace o2

#endif
