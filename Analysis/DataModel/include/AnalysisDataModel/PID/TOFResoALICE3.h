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

// Root includes
#include "TMath.h"
// O2 includes
#include "AnalysisDataModel/PID/ParamBase.h"

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
    const float ep = p * track.ErrorP;
    const float Lc = L / 29.9792458;

    /** perform PID **/
    const float mass2 = mass * mass;
    const float texp = Lc / p * TMath::Sqrt(mass2 + p2);
    const float etexp = Lc * mass2 / p2 / TMath::Sqrt(mass2 + p2) * ep;
    return TMath::Sqrt(etexp * etexp + mParameters[0] * mParameters[0] + evtimereso * evtimereso);
  }
  ClassDef(TOFResoALICE3, 1);
};

} // namespace o2::pid::tof

#endif
