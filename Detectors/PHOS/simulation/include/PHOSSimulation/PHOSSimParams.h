// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_PHOS_PHOSSIMPARAMS_H_
#define O2_PHOS_PHOSSIMPARAMS_H_

#include "SimConfig/ConfigurableParam.h"
#include "SimConfig/ConfigurableParamHelper.h"

namespace o2
{
namespace phos
{
// parameters used in responce calculation and digitization
// (mostly used in GEANT stepping and Digitizer)
struct PHOSSimParams : public o2::conf::ConfigurableParamHelper<PHOSSimParams> {
  //Parameters used in conversion of deposited energy to APD response
  float mLightYieldMean = 47000;                                  // Average number of photoelectrons per GeV
  float mIntrinsicAPDEfficiency = 0.02655;                        // APD efficiency including geometric coverage
  float mLightFactor = mLightYieldMean * mIntrinsicAPDEfficiency; // Average number of photons collected by APD per GeV deposited energy
  float mAPDFactor = (13.418 / mLightYieldMean / 100.) * 300.;    // factor relating light yield and APD response

  //Parameters used in electronic noise calculation and thresholds (Digitizer)
  bool mApplyTimeResolution = false; ///< Apply time resolution in digitization
  bool mApplyNonLinearity = false;   ///< Apply energy non-linearity in digitization
  bool mApplyDigitization = false;   ///< Apply energy digitization in digitization
  bool mApplyDecalibration = false;  ///< Apply de-calibration in digitization
  float mAPDNoise = 0.004;           ///< RMS of APD noise
  float mDigitThreshold = 2.5;       ///< minimal energy to keep digit
  float mADCwidth = 0.005;           ///< width of ADC channel in GeV
  float mTOFa = 0.5e-9;              ///< constant term of TOF resolution
  float mTOFb = 1.e-9;               ///< stohastic term of TOF resolution
  float mCellNonLineaityA = 0.;      ///< Amp of cel non-linearity
  float mCellNonLineaityB = 0.109;   ///< Energy scale of cel non-linearity
  float mCellNonLineaityC = 1.;      ///< Overall calibration

  float mZSthreshold = 0.005;    ///< Zero Suppression threshold
  float mTimeResolutionA = 2.;   ///< Time resolution parameter A (in ns)
  float mTimeResolutionB = 2.;   ///< Time resolution parameter B (in ns/GeV)
  float mTimeResThreshold = 0.5; ///< threshold for time resolution calculation (in GeV)
  float mMinNoiseTime = -200.;   ///< minimum time in noise channels (in ns)
  float mMaxNoiseTime = 2000.;   ///< minimum time in noise channels (in ns)

  O2ParamDef(PHOSSimParams, "PHOSSimParams");
};
} // namespace phos
} // namespace o2

#endif /* O2_PHOS_PHOSSIMPARAMS_H_ */
