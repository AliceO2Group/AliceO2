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

/// \class EMCALCalibInitParams
/// \brief  Init parameters for emcal calibrations
/// \author Joshua Koenig
/// \ingroup EMCALCalib
/// \since Apr 5, 2022

#ifndef EMCAL_CALIB_INIT_PARAMS_H_
#define EMCAL_CALIB_INIT_PARAMS_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace emcal
{

// class containing the parameters to trigger the calibrations
struct EMCALCalibParams : public o2::conf::ConfigurableParamHelper<EMCALCalibParams> {

  unsigned int minNEvents = 1e6;
  unsigned int minNEntries = 1e5;
  bool useNEventsForCalib = true;

  O2ParamDef(EMCALCalibParams, "EMCALCalibParams");
};

} // namespace emcal
} // namespace o2

#endif /*EMCAL_CALIB_INIT_PARAMS_H_ */