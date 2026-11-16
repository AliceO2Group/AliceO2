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
  float busyTimeNS() const { return readoutTimeNS + deadTimeNS; } ///< the time for which no new trigger can be received in nanoseconds
  // digitization settings
  int digithreads = 4;                                    ///< number of digitizer threads
  float maxMCStepSize = 0.1;                              ///< maximum size of MC steps
  bool doTR = true;                                       ///< switch for transition radiation
  SimParam::GasMixture gas = SimParam::GasMixture::Xenon; ///< the gas mixture in the TRD
  // ionisation in the gas
  int ionisationModel = -1;            ///< charge from 0: Geant4 energy deposit with PAI, 1: TRD collision tables, no PAI, -1: 1 with G4HepEm, else 0
  float ionisationMinBetaGamma = 0.04; ///< below this beta*gamma the charge comes from the energy deposit
  float ionisationMinEkin = 30e3;      ///< (eV) below this kinetic energy the charge comes from the energy deposit
  // TRF parameters
  int trf = 1;                  // Sampled TRF function. 0: default TRF, 1: TRF described in TRF TDR, 2: No TRF, 3: Landau dist as TRF (parameters specified below)
  float trf_landau_mu = 0.;     // Mu of the Landau distribution used to describe TRF
  float trf_landau_sigma = .03; // Sigma of the Landau distribution used to describe TRF
  float t0 = -0.170;            // Time delay before start of amplification peak, in us
  /// the ionisation model in use, resolving -1 by whether G4HepEm transports e-, e+ and gamma
  int activeIonisationModel(bool useG4HepEm) const { return ionisationModel >= 0 ? ionisationModel : (useG4HepEm ? 1 : 0); }
  O2ParamDef(TRDSimParams, "TRDSimParams");
};

} // namespace trd
} // namespace o2

#endif
