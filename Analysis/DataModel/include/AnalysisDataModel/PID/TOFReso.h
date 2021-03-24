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
/// \file   TOFReso.h
/// \author Nicolo' Jacazio
/// \since  07/08/2020
/// \brief  Implementation for the TOF PID response of the expected times resolution
///

#ifndef O2_ANALYSIS_PID_TOFRESO_H_
#define O2_ANALYSIS_PID_TOFRESO_H_

// O2 includes
#include "AnalysisDataModel/PID/ParamBase.h"
#include "ReconstructionDataFormats/PID.h"

namespace o2::pid::tof
{

class TOFReso : public Parametrization
{
 public:
  TOFReso() : Parametrization("TOFReso", 5){};
  ~TOFReso() override = default;
  float operator()(const float* x) const override
  {
    const float mom = abs(x[0]);
    if (mom <= 0) {
      return -999;
    }
    const float time = x[1];
    const float evtimereso = x[2];
    const float mass = x[3];
    const float dpp = mParameters[0] + mParameters[1] * mom + mParameters[2] * mass / mom; // mean relative pt resolution;
    const float sigma = dpp * time / (1. + mom * mom / (mass * mass));
    return sqrt(sigma * sigma + mParameters[3] * mParameters[3] / mom / mom + mParameters[4] * mParameters[4] + evtimereso * evtimereso);
  }
  ClassDef(TOFReso, 1);
};

float TOFResoParam(const float& momentum, const float& time, const float& evtimereso, const float& mass, const Parameters& parameters)
{
  if (momentum <= 0) {
    return -999;
  }
  const float dpp = parameters[0] + parameters[1] * momentum + parameters[2] * mass / momentum; // mean relative pt resolution;
  const float sigma = dpp * time / (1. + momentum * momentum / (mass * mass));
  return sqrt(sigma * sigma + parameters[3] * parameters[3] / momentum / momentum + parameters[4] * parameters[4] + evtimereso * evtimereso);
}

template <o2::track::PID::ID id, typename C, typename T>
float TOFResoParamTrack(const C& collision, const T& track, const Parameters& parameters)
{
  return TOFResoParam(track.p(), track.tofSignal(), collision.collisionTimeRes() * 1000.f, o2::track::pid_constants::sMasses[id], parameters);
}

} // namespace o2::pid::tof

#endif
