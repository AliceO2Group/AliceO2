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

#include "SimConfig/ConfigurableParam.h"
#include "SimConfig/ConfigurableParamHelper.h"

namespace o2
{
namespace cpv
{
// parameters used in responce calculation and digitization
// (mostly used in GEANT stepping and Digitizer)
struct CPVSimParams : public o2::conf::ConfigurableParamHelper<CPVSimParams> {
  //Parameters used in conversion of deposited energy to APD response
  int mnCellZ = 128;
  int mnCellX = 60;
  float mPadSizeZ = 1.13;       ///<  overall size of CPV active size
  float mPadSizeX = 2.1093;     ///<  in phi and z directions
  float mDetR = 0.1;            ///<  Relative energy fluctuation in track for 100 e-
  float mdEdx = 4.0;            ///<  Average energy loss in CPV;
  int mNgamz = 5;               ///<  Ionization size in Z
  int mNgamx = 9;               ///<  Ionization size in Phi
  float mCPVGasThickness = 1.3; ///<  width of ArC02 gas gap
  float mA = 1.0;               ///<  Parameter to model CPV response
  float mB = 0.7;               ///<  Parameter to model CPV response

  //Parameters used in electronic noise calculation and thresholds (Digitizer)
  bool mApplyDigitization = true;   ///< if energy digitization should be applied
  float mZSthreshold = 0.005;       ///< Zero Suppression threshold
  float mADCWidth = 0.005;          ///< Widht of ADC channel used for energy digitization
  float mNoise = 0.03;              ///<  charge noise in one pad
  float mCoeffToNanoSecond = 1.e+9; ///< Conversion for time units

  inline float CellWr() const { return mPadSizeX / 2.; } ///<  Distance between wires (2 wires above 1 pad)

  O2ParamDef(CPVSimParams, "CPVSimParams");
};
} // namespace cpv
} // namespace o2

#endif /* O2_CPV_CPVSIMPARAMS_H_ */
