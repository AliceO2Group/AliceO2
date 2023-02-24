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

#ifndef O2_CONF_DIGIPARAMS_H_
#define O2_CONF_DIGIPARAMS_H_

// Global parameters for TRD simulation / digitization

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include "TRDSimulation/SimParam.h"

namespace o2
{
namespace trd
{

// Global parameters for TRD simulation / digitization
/*
  See https://github.com/AliceO2Group/AliceO2/blob/dev/Common/SimConfig/doc/ConfigurableParam.md
*/
struct TRDSimParams : public o2::conf::ConfigurableParamHelper<TRDSimParams> {
  // Trigger parameters
  float readoutTimeNS = 3000;                    ///< the time the readout takes in ns (default 30 time bins = 3 us)
  float deadTimeNS = 11000;                      ///< trigger deadtime in ns (default 11 us)
  float busyTimeNS = readoutTimeNS + deadTimeNS; ///< the time for which no new trigger can be received in nanoseconds
  // digitization settings
  int digithreads = 4;                                    ///< number of digitizer threads
  float maxMCStepSize = 0.1;                              ///< maximum size of MC steps
  bool doTR = true;                                       ///< switch for transition radiation
  SimParam::GasMixture gas = SimParam::GasMixture::Xenon; ///< the gas mixture in the TRD
  O2ParamDef(TRDSimParams, "TRDSimParams");
};

} // namespace trd
} // namespace o2

#endif
