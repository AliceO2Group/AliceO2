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

#ifndef O2_TRD_CALIBRATIONS_H
#define O2_TRD_CALIBRATIONS_H

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  TRD calibration class for parameters which are saved frequently(/run)    //
//  2019 - Ported from various bits of AliRoot (SHTM)                        //
//  A wrapper api class to provide a user intuitive interface to the
//  underlying calibrations classes and ccdb and make life easy for all
//  bold statement, lets see if we get there.
//
//  Questions :
//  1. Its not clear to me yet what happens if I query an object that
//      is not defined as input in the dpl for the device we are using.
//
//  This pulls in functions from various subordinate classes like CalPad CalROC
//  CalDet etc. and renames them to clearly reflect their applicability,
//
//  This class does *not* store pointers to any subordinate objects as that
//  would imply caching. At each request the underlying object stored in
//  CCDB is sought, either directly or for that which is provided by DPL. The
//  latter still being unclear (17/09/2019)
//  BE AWARE THIS IS LIKELY TO CHANGE BEFORE START OF RUN 3 (25/09/2019)
///////////////////////////////////////////////////////////////////////////////

#include <memory>
#include <string>
#include "Rtypes.h"
#include "TRDBase/Geometry.h"
#include "TRDBase/ChamberCalibrations.h"
#include "TRDBase/ChamberStatus.h"
#include "TRDBase/PadCalibrationsAliases.h"
#include "TRDBase/CalOnlineGainTables.h"

namespace o2
{
namespace trd
{

class Calibrations
{

 public:
  Calibrations() = default;
  ~Calibrations() = default;
  //
  int getTimeStamp() const { return mTimeStamp; }
  void getCCDBObjects(long timestamp);
  void setOnlineGainTables(std::string& tablename);
  //
  float getVDrift(int roc, int col, int row) const;
  float getT0(int roc, int col, int row) const;
  float getExB(int roc) const;
  float getGainFactor(int roc, int col, int row) const;
  float getPadGainFactor(int roc, int col, int row) const;

  const PadStatus* getPadStatus() const { return mPadStatus; }
  const ChamberStatus* getChamberStatus() const { return mChamberStatus; }

  //online gain tables.
  float getOnlineGainAdcdac(int det, int row, int mcm) const;
  float getOnlineGainFGAN(int det, int row, int mcm, int adc) const;
  float getOnlineGainFGFN(int det, int row, int mcm, int adc) const;

 protected:
  long mTimeStamp; //run number of related to the current calibration.

  ChamberCalibrations* mChamberCalibrations; ///< from AliRoot: vDrift, T0, ExB and Gain for each chamber
  LocalVDrift* mLocalVDrift;                 ///< vDrift value per readout pad
  LocalT0* mLocalT0;                         ///< t0 value per readout pad
  LocalGainFactor* mLocalGainFactor;         ///< gain factor per readout pad
  PadNoise* mPadNoise;                       ///< noise value per readout pad
  ChamberStatus* mChamberStatus;             ///< status flag for each chamber
  PadStatus* mPadStatus;                     ///< status flag for each readout pad
  CalOnlineGainTables* mCalOnlineGainTables; ///< online gain table (obtained from Kr calibration)
  //
  ClassDefNV(Calibrations, 1);
};
} // namespace trd
} // namespace o2
#endif
