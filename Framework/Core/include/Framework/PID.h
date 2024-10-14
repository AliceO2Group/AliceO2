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

#ifndef o2_framework_PID_H_DEFINED
#define o2_framework_PID_H_DEFINED

#include <cmath>
#include "CommonConstants/PhysicsConstants.h"

///
/// \file PID.h
/// \author Nicolò Jacazio nicolo.jacazio@cern.ch
/// \since 2024-09-11
/// \brief TOF PID utilities to work with the information stored in the AO2D
///

namespace o2::framework::pid
{

namespace tof
{

/// @brief Compute the expected time of flight for a given momentum, length and massSquared
/// @param tofExpMom the expected momentum of the particle in GeV/c
/// @param length the track length in cm
/// @param massSquared the squared mass of the particle in GeV^2/c^4
/// @return the expected time of flight of the particle in ps
inline float MassToExpTime(float tofExpMom, float length, float massSquared)
{
  if (tofExpMom <= 0.f) {
    return -999.f;
  }
  return length * std::sqrt((massSquared) + (tofExpMom * tofExpMom)) / (o2::constants::physics::LightSpeedCm2PS * tofExpMom);
}

/// @brief Compute the signal of the time of flight for a given track time and expected time of flight
/// @param tracktime the measured time of flight (at the vertex) in ps
/// @param exptime the expected time of flight in ps
/// @return the signal of the time of flight
inline float TrackTimeToTOFSignal(float tracktime, float exptime)
{
  return tracktime * 1000.f + exptime;
}
} // namespace tof

} // namespace o2::framework::pid

#endif // o2_framework_PID_H_DEFINED
