// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_CPV_CPVSIMPARAMS_H_
#define O2_CPV_CPVSIMPARAMS_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace cpv
{
// parameters used in responce calculation and digitization
// (mostly used in GEANT stepping and Digitizer)
struct CPVSimParams : public o2::conf::ConfigurableParamHelper<CPVSimParams> {

  std::string mCCDBPath = "localtest"; ///< use "localtest" to avoid connecting ccdb server, otherwise use ccdb-test.cern.ch

  //Parameters used in conversion of deposited energy to APD response
  int mnCellX = 128;
  int mnCellZ = 60;
  float mPadSizeX = 1.13;       ///<  overall size of CPV active size
  float mPadSizeZ = 2.1093;     ///<  in phi and z directions
  float mDetR = 0.1;            ///<  Relative energy fluctuation in track for 100 e-
  float mdEdx = 4.0;            ///<  Average energy loss in CPV;
  int mNgamz = 5;               ///<  Ionization size in Z
  int mNgamx = 9;               ///<  Ionization size in Phi
  float mCPVGasThickness = 1.3; ///<  width of ArC02 gas gap
  float mA = 1.0;               ///<  Parameter to model CPV response
  float mB = 0.7;               ///<  Parameter to model CPV response

  //Parameters used in electronic noise calculation and thresholds (Digitizer)
  float mReadoutTime = 5.;          ///< Read-out time in ns for default simulaionts
  float mDeadTime = 20.;            ///< PHOS dead time (includes Read-out time i.e. mDeadTime>=mReadoutTime)
  float mReadoutTimePU = 2000.;     ///< Read-out time in ns if pileup simulation on in DigitizerSpec
  float mDeadTimePU = 30000.;       ///< PHOS dead time if pileup simulation on in DigitizerSpec
  bool mApplyDigitization = true;   ///< if energy digitization should be applied
  float mZSthreshold = 0.01;        ///< Zero Suppression threshold
  float mADCWidth = 0.005;          ///< Widht of ADC channel used for energy digitization
  float mNoise = 0.01;              ///<  charge noise in one pad
  float mCoeffToNanoSecond = 1.e+9; ///< Conversion for time units
  float mSortingDelta = 0.1;        ///< used in sorting clusters inverse sorting band in cm

  //Parameters used in clusterization
  float mDigitMinEnergy = 0.01;       ///< Minimal amplitude of a digit to be used in cluster
  float mClusteringThreshold = 0.050; ///< Seed digit minimal amplitude
  float mUnfogingEAccuracy = 1.e-3;   ///< Accuracy of energy calculation in unfoding prosedure (GeV)
  float mUnfogingXZAccuracy = 1.e-1;  ///< Accuracy of position calculation in unfolding procedure (cm)
  float mLocalMaximumCut = 0.030;     ///< Threshold to separate local maxima
  float mLogWeight = 4.5;             ///< weight in cluster center of gravity calculation
  int mNMaxIterations = 10;           ///< Maximal number of iterations in unfolding procedure
  bool mUnfoldClusters = false;       ///< Perform cluster unfolding?

  inline float CellWr() const { return 0.5 * mPadSizeX; } ///<  Distance between wires (2 wires above 1 pad)

  O2ParamDef(CPVSimParams, "CPVSimParams");
};
} // namespace cpv
} // namespace o2

#endif /* O2_CPV_CPVSIMPARAMS_H_ */
