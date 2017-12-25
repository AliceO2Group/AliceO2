// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Propagator
/// \brief Singleton class for track propagation routines
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_BASE_PROPAGATOR_
#define ALICEO2_BASE_PROPAGATOR_

#include "CommonConstants/PhysicsConstants.h"
#include "DetectorsBase/Track.h"

namespace o2
{
namespace field
{
class MagFieldFast;
}

namespace Base
{
namespace Track
{
class Propagator
{
 public:
  static Propagator* Instance()
  {
    static Propagator instance;
    return &instance;
  }

  bool PropagateToXBxByBz(o2::Base::Track::TrackParCov& track, float x,
                          float mass = o2::constants::physics::MassPionCharged, float maxSnp = 0.85,
                          float maxStep = 2.0, int matCorr = 1, int signCorr = 0);

  Propagator(Propagator const&) = delete;
  Propagator(Propagator&&) = delete;
  Propagator& operator=(Propagator const&) = delete;
  Propagator& operator=(Propagator&&) = delete;

 private:
  Propagator();
  ~Propagator() = default;

  const o2::field::MagFieldFast* mField = nullptr;

  ClassDef(Propagator, 0);
};
}
}
}

#endif
