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
/// \file   BetheBloch.h
/// \author Nicolo' Jacazio
/// \since  07/08/2020
/// \brief  Implementation for the TPC PID response of the BB parametrization
///

#ifndef O2_ANALYSIS_PID_BETHEBLOCH_H_
#define O2_ANALYSIS_PID_BETHEBLOCH_H_

#include "TPCSimulation/Detector.h"
#include "AnalysisDataModel/PID/ParamBase.h"
#include "ReconstructionDataFormats/PID.h"

namespace o2::pid::tpc
{

class BetheBloch : public Parametrization
{
 public:
  BetheBloch() : Parametrization("BetheBloch", 7){};
  ~BetheBloch() override = default;
  float operator()(const float* x) const override
  {
    return mParameters[5] * o2::tpc::Detector::BetheBlochAleph(x[0], mParameters[0], mParameters[1], mParameters[2], mParameters[3], mParameters[4]) * TMath::Power(x[1], mParameters[6]);
  }
  ClassDef(BetheBloch, 1);
};

float BetheBlochParam(const float& momentum, const float& mass, const float& charge, const Parameters& parameters)
{
  return parameters[5] * o2::tpc::Detector::BetheBlochAleph(momentum / mass, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]) * pow(charge, parameters[6]);
}

template <o2::track::PID::ID id, typename T>
float BetheBlochParamTrack(const T& track, const Parameters& parameters)
{
  return BetheBlochParam(track.tpcInnerParam(), o2::track::pid_constants::sMasses[id], (float)o2::track::pid_constants::sCharges[id], parameters);
}

} // namespace o2::pid::tpc

#endif
