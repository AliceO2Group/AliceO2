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
/// \file   BetheBloch.h
/// \author Nicolo' Jacazio
/// \since  07/08/2020
/// \brief  Implementation for the TPC PID response of the BB parametrization
///

#ifndef O2_ANALYSIS_PID_BETHEBLOCH_H_
#define O2_ANALYSIS_PID_BETHEBLOCH_H_

#include "TPCSimulation/Detector.h"
#include "AnalysisDataModel/PID/ParamBase.h"

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

} // namespace o2::pid::tpc

#endif
