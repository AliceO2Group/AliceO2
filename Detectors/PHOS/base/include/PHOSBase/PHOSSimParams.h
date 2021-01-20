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

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace phos
{
// parameters used in responce calculation and digitization
// (mostly used in GEANT stepping and Digitizer)
struct PHOSSimParams : public o2::conf::ConfigurableParamHelper<PHOSSimParams> {

  std::string mCCDBPath = "localtest"; ///< use "localtest" to avoid connecting ccdb server, otherwise use ccdb-test.cern.ch

  //Parameters used in conversion of deposited energy to APD response
  float mLightYieldMean = 47000;                                  // Average number of photoelectrons per GeV
  float mIntrinsicAPDEfficiency = 0.02655;                        // APD efficiency including geometric coverage
  float mLightFactor = mLightYieldMean * mIntrinsicAPDEfficiency; // Average number of photons collected by APD per GeV deposited energy
  float mAPDFactor = (13.418 / mLightYieldMean / 100.) * 300.;    // factor relating light yield and APD response

  //Parameters used in electronic noise calculation and thresholds (Digitizer)
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

  short mZSthreshold = 1;        ///< Zero Suppression threshold
  float mTimeResolutionA = 2.;   ///< Time resolution parameter A (in ns)
  float mTimeResolutionB = 2.;   ///< Time resolution parameter B (in ns/GeV)
  float mTimeResThreshold = 0.5; ///< threshold for time resolution calculation (in GeV)
  float mMinNoiseTime = -200.;   ///< minimum time in noise channels (in ns)
  float mMaxNoiseTime = 2000.;   ///< minimum time in noise channels (in ns)

  //Parameters used in Raw simulation
  float mSampleDecayTime = 0.091; ///< Time parameter in Gamma2 function (1/tau, 100.e-9/2.1e-6)

  // //Parameters used in raw data reconstruction
  short mSpikeThreshold = 100; ///< Single spike >100 ADC channels
  short mBaseLine = 0;         ///<
  short mPreSamples = 2;       ///< number of pre-samples readout before sample (if no pedestal subtrauction)
  short mMCOverflow = 970;     ///< Overflow level for MC simulations: 1023-(pedestal~50)
  float mTimeTick = 100.;      ///< ns to PHOS digitization step conversion

  // bool  mSubtractPedestal = false ;    ///< subtract pedestals
  // bool  mCreateSampleQualityOutput = false ; ///< Create stream of sample quality
  // bool  mApplyBadMap = false ;         ///< Apply bad map in sample fitting
  // short mChiMinCut = 0 ;               ///< Minimal cut on sample quality
  // short mChiMaxCut = 1000;             ///< Maximal cut on sample quality
  // std::string mFitterVersion = "default"; ///< version of raw fitter to be used

  //Parameters used in clusterization
  float mLogWeight = 4.5;             ///< Cutoff used in log. weight calculation
  float mDigitMinEnergy = 0.010;      ///< Minimal energy of digits to be used in cluster (GeV)
  float mClusteringThreshold = 0.050; ///< Minimal energy of digit to start clustering (GeV)
  float mLocalMaximumCut = 0.015;     ///< Minimal height of local maximum over neighbours
  bool mUnfoldClusters = false;       ///< To perform cluster unfolding
  float mUnfogingEAccuracy = 1.e-3;   ///< Accuracy of energy calculation in unfoding prosedure (GeV)
  float mUnfogingXZAccuracy = 1.e-1;  ///< Accuracy of position calculation in unfolding procedure (cm)
  int mNMaxIterations = 10;           ///< Maximal number of iterations in unfolding procedure
  int mNLMMax = 30;                   ///< Maximal number of local maxima in unfolding
  float mCoreR = 3.5;                 ///< Radius to caluclate core energy
  float mSortingDelta = 1.;           ///< used in sorting clusters

  O2ParamDef(PHOSSimParams, "PHOSSimParams");
};
} // namespace phos
} // namespace o2

#endif /* O2_PHOS_PHOSSIMPARAMS_H_ */
