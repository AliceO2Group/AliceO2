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
/// \file   TPCReso.h
/// \author Nicolo' Jacazio
/// \since  07/08/2020
/// \brief  Implementation for the TPC PID response of the BB resolution
///

#ifndef O2_ANALYSIS_PID_TPCRESO_H_
#define O2_ANALYSIS_PID_TPCRESO_H_

#include "AnalysisDataModel/PID/ParamBase.h"
#include "ReconstructionDataFormats/PID.h"

namespace o2::pid::tpc
{

class TPCReso : public Parametrization
{
 public:
  TPCReso() : Parametrization("TPCReso", 2){};
  ~TPCReso() override = default;
  float operator()(const float* x) const override
  {
    // relative dEdx resolution rel sigma = fRes0*sqrt(1+fResN2/npoint)
    return x[0] * mParameters[0] * (x[1] > 0 ? sqrt(1. + mParameters[1] / x[1]) : 1.f);
  }
  ClassDef(TPCReso, 1);
};

float TPCResoParam(const float& signal, const float& npoints, const Parameters& parameters)
{
  return signal * parameters[0] * (npoints > 0 ? sqrt(1. + parameters[1] / npoints) : 1.f);
}

template <o2::track::PID::ID id, typename T>
float TPCResoParamTrack(const T& track, const Parameters& parameters)
{
  return TPCResoParam(track.tpcSignal(), (float)track.tpcNClsFound(), parameters);
}

} // namespace o2::pid::tpc

#endif
