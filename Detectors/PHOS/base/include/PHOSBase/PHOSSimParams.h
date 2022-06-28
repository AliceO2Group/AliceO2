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

#ifndef O2_PHOS_PHOSSIMPARAMS_H_
#define O2_PHOS_PHOSSIMPARAMS_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace phos
{
// parameters used in responce calculation and digitization
// (mostly used in GEANT stepping and Digitizer)
struct PHOSSimParams : public o2::conf::ConfigurableParamHelper<PHOSSimParams> {

  std::string mCCDBPath = "ccdb"; ///< use "localtest" to avoid connecting ccdb server, otherwise use ccdb-test.cern.ch

  // Parameters used in conversion of deposited energy to APD response
  float mLightYieldPerGeV = 526.;                 ///< Average number of photoelectrons per GeV: 1.983 gamma/MeV * 0.2655 PDE eff of APD
  std::string mDigitizationCalibPath = "default"; ///< use "default" to use default calibration or use ccdb.cern.ch
  std::string mDigitizationTrigPath = "default";  ///< use "default" to use default map and turn-on or use ccdb.cern.ch

  // Parameters used in electronic noise calculation and thresholds (Digitizer)
  float mReadoutTime = 5.;           ///< Read-out time in ns for default simulaionts
  float mDeadTime = 20.;             ///< PHOS dead time (includes Read-out time i.e. mDeadTime>=mReadoutTime)
  float mReadoutTimePU = 2000.;      ///< Read-out time in ns if pileup simulation on in DigitizerSpec
  float mDeadTimePU = 30000.;        ///< PHOS dead time if pileup simulation on in DigitizerSpec
  bool mApplyTimeResolution = false; ///< Apply time resolution in digitization
  bool mApplyNonLinearity = false;   ///< Apply energy non-linearity in digitization
  bool mApplyDigitization = false;   ///< Apply energy digitization in digitization
  float mAPDNoise = 0.005;           ///< RMS of APD noise
  float mDigitThreshold = 2.;        ///< minimal energy to keep digit in ADC counts
  float mADCwidth = 0.005;           ///< width of ADC channel in GeV
  float mTOFa = 0.5e-9;              ///< constant term of TOF resolution
  float mTOFb = 1.e-9;               ///< stohastic term of TOF resolution
  float mCellNonLineaityA = 0.;      ///< Amp of cel non-linearity
  float mCellNonLineaityB = 0.109;   ///< Energy scale of cel non-linearity
  float mCellNonLineaityC = 1.;      ///< Overall calibration

  short mZSthreshold = 1;         ///< Zero Suppression threshold
  float mTimeResolutionA = 2.e-9; ///< Time resolution parameter A (in sec)
  float mTimeResolutionB = 2.e-9; ///< Time resolution parameter B (in sec/GeV)
  float mTimeResThreshold = 0.5;  ///< threshold for time resolution calculation (in GeV)
  float mMinNoiseTime = -200.e-9; ///< minimum time in noise channels (in sec)
  float mMaxNoiseTime = 2000.e-9; ///< minimum time in noise channels (in sec)

  float mTrig2x2MinThreshold = 800.; ///< threshold to simulate 2x2 trigger turn-on curve (in ADC counts~0.005 GeV/count!)
  float mTrig4x4MinThreshold = 900.; ///< threshold to simulate 4x4 trigger turn-on curve (in ADC counts!)

  // Parameters used in Raw simulation
  float mSampleDecayTime = 0.091; ///< Time parameter in Gamma2 function (1/tau, 100.e-9/2.1e-6)

  // //Parameters used in raw data reconstruction
  short mSpikeThreshold = 100;          ///< Single spike >100 ADC channels
  short mBaseLine = 0;                  ///<
  short mPreSamples = 2;                ///< number of pre-samples readout before sample (if no pedestal subtrauction)
  short mMCOverflow = 970;              ///< Overflow level for MC simulations: 1023-(pedestal~50)
  float mTimeTick = 100.;               ///< ns to PHOS digitization step conversion
  float mTRUTimeTick = 25.;             ///< ns to PHOS TRU digitization step
  float mSampleTimeFitAccuracy = 1.e-3; // Abs accuracy of time fit of saturated samples (in 100ns tick units)
  float mSampleAmpFitAccuracy = 1.e-2;  // Relative accuracy of amp. fit
  short mNIterations = 5;               ///< maximal number of iterations in oveflow sample fit

  // bool  mSubtractPedestal = false ;    ///< subtract pedestals
  // bool  mCreateSampleQualityOutput = false ; ///< Create stream of sample quality
  // bool  mApplyBadMap = false ;         ///< Apply bad map in sample fitting
  // short mChiMinCut = 0 ;               ///< Minimal cut on sample quality
  // short mChiMaxCut = 1000;             ///< Maximal cut on sample quality
  // std::string mFitterVersion = "default"; ///< version of raw fitter to be used

  // Parameters used in clusterization
  float mLogWeight = 4.5;              ///< Cutoff used in log. weight calculation
  float mDigitMinEnergy = 0.010;       ///< Minimal energy of digits to be used in cluster (GeV)
  float mClusteringThreshold = 0.050;  ///< Minimal energy of digit to start clustering (GeV)
  float mLocalMaximumCut = 0.015;      ///< Minimal height of local maximum over neighbours
  int mUnfoldMaxSize = 100;            ///< maximal number of cells in cluster to be unfolded
  bool mUnfoldClusters = true;         ///< To perform cluster unfolding
  float mUnfogingEAccuracy = 1.e-2;    ///< Accuracy of energy calculation in unfoding prosedure (GeV)
  float mUnfogingXZAccuracy = 1.e-1;   ///< Accuracy of position calculation in unfolding procedure (cm)
  float mUnfogingChi2Accuracy = 1.e-2; ///< critical chi2/NDF
  int mNMaxIterations = 10;            ///< Maximal number of iterations in unfolding procedure
  float mCoreR = 3.5;                  ///< Radius to caluclate core energy
  float mSortingDelta = 1.;            ///< used in sorting clusters

  O2ParamDef(PHOSSimParams, "PHOSSimParams");
};
} // namespace phos
} // namespace o2

#endif /* O2_PHOS_PHOSSIMPARAMS_H_ */
