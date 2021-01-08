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
/// \file   TPCReso.h
/// \author Nicolo' Jacazio
/// \since  07/08/2020
/// \brief  Implementation for the TPC PID response of the BB resolution
///

#ifndef O2_ANALYSIS_PID_TPCRESO_H_
#define O2_ANALYSIS_PID_TPCRESO_H_

#include "AnalysisDataModel/PID/ParamBase.h"

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

} // namespace o2::pid::tpc

#endif
