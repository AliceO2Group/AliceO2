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

#ifndef O2_CPV_CPVCALIBPARAMS_H_
#define O2_CPV_CPVCALIBPARAMS_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include "CommonUtils/NameConf.h"

namespace o2
{
namespace cpv
{
// parameters used in responce calculation and digitization
// (mostly used in GEANT stepping and Digitizer)
struct CPVCalibParams : public o2::conf::ConfigurableParamHelper<CPVCalibParams> {

  // Parameters used in pedestal calibration
  uint16_t pedToleratedGapWidth = 5;    ///< Tolerated gap between bins: if |bin1 - bin2| < width -> bin1 and bin2 belongs to same peak
  uint32_t pedMinEvents = 100;          ///< Minimal number of events to produce calibration
  float pedSuspiciousPedestalRMS = 10.; ///< Take additional care for channel if its RMS >  mPedSuspiciousPedestalRMS
  float pedZSnSigmas = 3.;              ///< Zero Suppression threshold

  // Parameters used in noise scan calibration
  uint32_t noiseMinEvents = 100;                    ///< Minimal number of events to produce calibration
  float noiseToleratedChannelEfficiencyLow = 0.9;   ///< Tolerated channel efficiency (lower limit)
  float noiseToleratedChannelEfficiencyHigh = 1.01; ///< Tolerated channel efficiency (upper limit)
  uint16_t noiseThreshold = 10;                     ///< ADC threshold
  float noiseFrequencyCriteria = 0.5;               ///< Appearance frequency of noisy channels

  // Parameters used in gain calibration
  uint32_t gainMinEvents = 1000;                  ///< Minimal number of events to produce calibration in one channel
  uint32_t gainMinNChannelsToCalibrate = 2000;    ///< Minimal number of channels ready to be calibrated to produce calibration
  float gainDesiredLandauMPV = 200.;              ///< Desired LandauMPV of the spectrum: gain coeff = 200./(max Ampl of the cluster)
  float gainToleratedChi2PerNDF = 100.;           ///< Tolerated max Chi2 of the fit
  float gainMinAllowedCoeff = 0.1;                ///< Min value of gain coeff
  float gainMaxAllowedCoeff = 10.;                ///< Max value of gain coeff
  float gainFitRangeL = 10.;                      ///< Fit range of amplitude spectrum (left)
  float gainFitRangeR = 1000.;                    ///< Fit range of amplitude spectrum (right)
  unsigned char gainMinClusterMultForCalib = 3;   ///< Min cluster multiplicity for calibration digit production
  unsigned char gainMaxClusterMultForCalib = 15;  ///< Max cluster multiplicity for calibration digit production
  uint32_t gainCheckForCalibrationInterval = 100; ///< Check interval (in TFs) if statistics is enough

  O2ParamDef(CPVCalibParams, "CPVCalibParams");
};
} // namespace cpv

namespace framework
{
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::cpv::CPVCalibParams> : std::true_type {
};
} // namespace framework

} // namespace o2

#endif /* O2_CPV_CPVCALIBPARAMS_H_ */
