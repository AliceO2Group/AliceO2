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

/// \file FV0DigParam.h
/// \brief Configurable digitization parameters

#ifndef ALICEO2_FV0_DIG_PARAM
#define ALICEO2_FV0_DIG_PARAM

#include "CommonConstants/PhysicsConstants.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2::fv0
{
// parameters of FV0 digitization / transport simulation
struct FV0DigParam : o2::conf::ConfigurableParamHelper<FV0DigParam> {
  float photoCathodeEfficiency = 0.23;                  // quantum efficiency = nOfPhotoE_emitted_by_photocathode / nIncidentPhotons
  float lightYield = 0.01;                              // light collection efficiency to be tuned using collision data [1%]
  float adcChannelsPerMip = 16;                         // Default: 16 for pp and 8 for PbPb
  float getChannelsPerMilivolt() const { return adcChannelsPerMip / 7.5; } // Non-trivial conversion depending on the pulseshape: amplitude to charge
  float chargeThrForMeanTime = 10;                      // Charge threshold, only above which the time is taken into account in calculating the mean time of all qualifying channels

  /// Parameters for the FV0 waveform [Conv. of expo. with Landau]
  // For ring 1-4
  float offsetRingA1ToA4 = 15.87e-09;
  float normRingA1ToA4base = 7.9061033e-13;
  float getNormRingA1ToA4() const { return normRingA1ToA4base * adcChannelsPerMip / 16; }
  float constRingA1ToA4 = -25.6165;
  float slopeRingA1ToA4 = 4.7942e+08;
  float mpvRingA1ToA4 = -6.38203e-08;
  float sigmaRingA1ToA4 = 2.12167e-09;
  // For ring 5
  float offsetRing5 = 16.38e-09;
  float normRing5base = 8.0303587e-13;
  float getNormRing5() const { return normRing5base * adcChannelsPerMip / 16; }
  float constRing5 = -66.76;
  float slopeRing5 = 9.43117e+08;
  float mpvRing5 = -6.44167e-08;
  float sigmaRing5 = 2.3621e-09;

  float timeShiftCfd = 5.3;                                                      // TODO: adjust after FV0 with FEE measurements are done
  float singleMipThreshold = 3.0;                                                // in [MeV] of deposited energy
  float singleHitTimeThreshold = 120.0;                                          // in [ns] to skip very slow particles
  UInt_t waveformNbins = 10000;                                                  // number of bins for the analog pulse waveform
  float waveformBinWidth = 0.01302;                                              // bin width [ns] for analog pulse waveform
  float avgCfdTimeForMip = 8.63;                                                 // in ns to shift the CFD time to zero TODO [do ring wise]
  bool isIntegrateFull = false;                                                  // Full charge integration widow in 25 ns
  float cfdCheckWindow = 2.5;                                                    // time window for the cfd in ns to trigger the charge integration
  int avgNumberPhElectronPerMip = 201;                                           // avg number of photo-electrons per MIP
  float globalTimeOfFlight = 315.0 / o2::constants::physics::LightSpeedCm2NS;    // TODO [check the correct value for distance of FV0 to IP]
  float mCfdDeadTime = 15.6;                                                     // [ns]
  float mCFD_trsh = 3.;                                                          // [mV]
  float getCFDTrshInAdc() const { return mCFD_trsh * getChannelsPerMilivolt(); } // [ADC channels]
  /// Parameters for trigger simulation
  bool useMaxChInAdc = true;         // default = true
  int adcChargeCenThr = 3 * 498;     // threshold value of ADC charge for Central trigger
  int adcChargeSCenThr = 1 * 498;    // threshold value of ADC charge for Semi-central trigger
  int maxCountInAdc = 4095;          // to take care adc ADC overflow

  short mTime_trg_gate = 153; // #channels as in TCM as in Pilot beams ('OR gate' setting in TCM tab in ControlServer)

  O2ParamDef(FV0DigParam, "FV0DigParam");
};
} // namespace o2::fv0

#endif
