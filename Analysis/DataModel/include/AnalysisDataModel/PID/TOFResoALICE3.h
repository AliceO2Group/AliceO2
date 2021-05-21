// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file   TOFResoALICE3.h
/// \author Nicolo' Jacazio
/// \since  11/03/2021
/// \brief  Implementation for the TOF PID response of the expected times resolution
///

#ifndef O2_ANALYSIS_PID_TOFRESOALICE3_H_
#define O2_ANALYSIS_PID_TOFRESOALICE3_H_

// O2 includes
#include "AnalysisDataModel/PID/ParamBase.h"
#include "ReconstructionDataFormats/PID.h"

namespace o2::pid::tof
{

class TOFResoALICE3 : public Parametrization
{
 public:
  TOFResoALICE3() : Parametrization("TOFResoALICE3", 1){};
  ~TOFResoALICE3() override = default;
  float operator()(const float* x) const override
  {
    const float p = abs(x[0]);
    if (p <= 0) {
      return -999;
    }

    /** get info **/
    const float time = x[1];
    const float evtimereso = x[2];
    const float mass = x[3];
    const float L = x[4];
    const float p2 = p * p;
    // const float ep = x[5] * x[6];
    const float ep = x[5] * p2;
    const float Lc = L / 0.0299792458f;
    // const float Lc = L / 29.9792458f;

    const float mass2 = mass * mass;
    const float etexp = Lc * mass2 / p2 / sqrt(mass2 + p2) * ep;
    return sqrt(etexp * etexp + mParameters[0] * mParameters[0] + evtimereso * evtimereso);
  }
  ClassDef(TOFResoALICE3, 1);
};

float TOFResoALICE3Param(const float& momentum, const float& momentumError, const float& evtimereso, const float& length, const float& mass, const Parameters& parameters)
{
  if (momentum <= 0) {
    return -999;
  }

  const float p2 = momentum * momentum;
  const float Lc = length / 0.0299792458f;
  const float mass2 = mass * mass;
  const float ep = momentumError * momentum;
  // const float ep = momentumError * p2;
  const float etexp = Lc * mass2 / p2 / sqrt(mass2 + p2) * ep;
  return sqrt(etexp * etexp + parameters[0] * parameters[0] + evtimereso * evtimereso);
}

template <o2::track::PID::ID id, typename C, typename T>
float TOFResoALICE3ParamTrack(const C& collision, const T& track, const Parameters& parameters)
{
  const float BETA = tan(0.25f * static_cast<float>(M_PI) - 0.5f * atan(track.tgl()));
  const float sigmaP = sqrt(pow(track.pt(), 2) * pow(track.sigma1Pt(), 2) + (BETA * BETA - 1.f) / (BETA * (BETA * BETA + 1.f)) * (track.tgl() / sqrt(track.tgl() * track.tgl() + 1.f) - 1.f) * pow(track.sigmaTgl(), 2));
  // const float sigmaP = std::sqrt( track.getSigma1Pt2() ) * track.pt();
  return TOFResoALICE3Param(track.p(), sigmaP, collision.collisionTimeRes() * 1000.f, track.length(), o2::track::pid_constants::sMasses[id], parameters);
  // return TOFResoALICE3Param(track.p(), track.sigma1Pt(), collision.collisionTimeRes() * 1000.f, track.length(), o2::track::pid_constants::sMasses[id], parameters);
}

} // namespace o2::pid::tof

#endif
