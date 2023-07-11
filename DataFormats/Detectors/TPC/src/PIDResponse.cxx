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
/// @file   PIDResponse.cxx
/// @author Tuba Gündem, tuba.gundem@cern.ch
///

#include "DataFormatsTPC/PIDResponse.h"
#include "DataFormatsTPC/BetheBlochAleph.h"

using namespace o2::tpc;

double PIDResponse::getExpectedSignal(const TrackTPC& track, const o2::track::PID::ID id) const
{
  const double bg = static_cast<double>(track.getP() / o2::track::pid_constants::sMasses[id]);
  if (bg < 0.05) {
    return -999.;
  }
  const double bethe = mMIP * o2::tpc::BetheBlochAleph(bg, mBetheBlochParams[0], mBetheBlochParams[1], mBetheBlochParams[2], mBetheBlochParams[3], mBetheBlochParams[4]) * std::pow(static_cast<double>(o2::track::pid_constants::sCharges[id]), mChargeFactor);
  return bethe >= 0. ? bethe : -999.;
}

// get most probable PID
o2::track::PID::ID PIDResponse::getMostProbablePID(const TrackTPC& track) const
{
  const double dEdx = track.getdEdx().dEdxTotTPC;

  if (dEdx < 10) {
    return o2::track::PID::Pion;
  }

  auto id = o2::track::PID::Electron;
  double distanceMin = std::abs(dEdx - getExpectedSignal(track, id));

  // calculate the distance to the expected dEdx signals
  // start from Pion to exlude Muons
  for (o2::track::PID::ID i = o2::track::PID::Pion; i < o2::track::PID::NIDs; i++) {
    const double distance = std::abs(dEdx - getExpectedSignal(track, i));
    if (distance < distanceMin) {
      id = i;
      distanceMin = distance;
    }
  }

  return id;
}
