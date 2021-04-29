// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FV0_SIMPARAMS_H_
#define O2_FV0_SIMPARAMS_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include "CommonConstants/PhysicsConstants.h"

namespace o2
{
namespace fv0
{
// parameters of FV0 digitization / transport simulation
struct FV0DigParam : public o2::conf::ConfigurableParamHelper<FV0DigParam> {
  // NOTUSED float intrinsicTimeRes = 0.91;       // time resolution
  float photoCathodeEfficiency = 0.23; // quantum efficiency = nOfPhotoE_emitted_by_photocathode / nIncidentPhotons
  float lightYield = 0.01;             // light collection efficiency to be tuned using collision data [1%]
  // NOTUSED float pmtGain = 5e4;                 // value for PMT R5924-70 at default FV0 gain
  // NOTUSED float pmtTransitTime = 9.5;          // PMT response time (corresponds to 1.9 ns rise time)
  // NOTUSED float pmtTransparency = 0.25;        // Transparency of the first dynode of the PMT
  float shapeConst = 1.18059e-14;  // Crystal ball const parameter
  float shapeMean = 9.5;           // Crystal ball mean  parameter
  float shapeAlpha = -6.56586e-01; // Crystal ball alpha parameter
  float shapeN = 2.36408e+00;      // Crystal ball N     parameter
  float shapeSigma = 3.55445;      // Crystal ball sigma parameter
  //float timeShiftCfd = 3.3;          // From the cosmic measurements of FV0 [keep it for reference]
  float timeShiftCfd = 5.3;                                                   // TODO: adjust after FV0 with FEE measurements are done
  float singleMipThreshold = 3.0;                                             // in [MeV] of deposited energy
  float singleHitTimeThreshold = 120.0;                                       // in [ns] to skip very slow particles
  UInt_t waveformNbins = 10000;                                               // number of bins for the analog pulse waveform
  float waveformBinWidth = 0.01302;                                           // bin width [ns] for analog pulse waveform
  float avgCfdTimeForMip = 8.63;                                              // in ns to shift the CFD time to zero TODO do ring wise
  bool isIntegrateFull = false;                                               // Full charge integration widow in 25 ns
  float cfdCheckWindow = 2.5;                                                 // time window for the cfd in ns to trigger the charge integration
  int avgNumberPhElectronPerMip = 201;                                        // avg number of photo-electrons per MIP
  float globalTimeOfFlight = 315.0 / o2::constants::physics::LightSpeedCm2NS; //TODO check the correct value for distance of FV0 to IP
  float mCFDdeadTime = 15.6;                                                  // ns
  float mCFD_trsh = 3.;                                                       // [mV]
  ///Parameters for trigger simulation
  int adcChargeHighMultTh = 3 * 498; //threshold value of ADC charge for high multiplicity trigger

  O2ParamDef(FV0DigParam, "FV0DigParam");
};
} // namespace fv0
} // namespace o2

#endif
