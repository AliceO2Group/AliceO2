// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include <TMath.h>
#include <TH1F.h>
#include <TH2F.h>
#include <sstream>

#include "TRDBase/TRDGeometry.h"
#include "TRDBase/Calibrations.h"
#include "TRDBase/ChamberCalibrations.h"
#include "TRDBase/LocalVDrift.h"
#include "TRDBase/LocalT0.h"
#include "TRDBase/LocalGainFactor.h"
//#include "TRDBase/TrapConfig.h"
//#include "TRDBase/PRFWidth.h"
#include "fairlogger/Logger.h"

//#include <CCDB/TObjectWrapper.h>
#include "CCDB/Condition.h"
#include "CCDB/Manager.h"

using namespace o2::trd;

// THIS IS A WIP !  16/09/2019
//

// first lets port the functions from CalROC :
// This therefore includes CalPad (Local[VDrift,T0,GainFactor,PRFWidth,PadNoise)
// TODO will come back to this in a while, more pressing issues for now.
//      This is here mostly as a stub to remember how to do it.
//
/* double Calibrations::getVDriftMean(int run) const
{
  //
  // Calculate the mean
  //
 // CCDB

    
  auto ccdb = o2::ccdb::Manager::Instance();
  ccdb->setDefaultStorage("local://O2CDB");


  auto chambercalibrationscond = ccdb->getCondition("TRD/ChamberCalibrations", run);
  ChamberCalibrations *chambercalibrations=nullptr;
  chambercalibrationscond->getObjectAs(chambercalibrations);
  auto localvdriftcond = ccdb->getCondition("TRD/LocalVDrift",run);
  LocalVDrift *localvdrifts=nullptr;
  localvdriftcond->getObjectAs(localvdrifts);

      std::vector<double> ddata(540);
  int nPoints = 0;
  //loop over localvdrifts and for each chamber apply the chambers vdrift and the take mean.
  for (int i = 0; i < 540; i++) {
      ddata[nPoints] = (double)chambercalibrations->getVDrift(i);
      nPoints++;
    }
  double mean = TMath::Mean(nPoints, ddata.data());
  return mean;
}*/

/* double Calibrations::getVDrift(int run)
{
      //
       auto ccdb = o2::ccdb::Manager::Instance();
       ccdb->setDefaultStorage("local://O2CDB");
 

       auto chambercalibrationscond = ccdb->getCondition("TRD/ChamberCalibrations", run);
       ChamberCalibrations *chambercalibrations=nullptr;
       chambercalibrationscond->getObjectAs(chambercalibrations);
       auto localvdriftcond = ccdb->getCondition("TRD/LocalVDrift",run);
       LocalVDrift *localvdrifts=nullptr;
       localvdriftcond->getObjectAs(localvdrifts);

       return (double)chambercalibrations->getVDrift(det)* (double)localvdrifts->getValue(det,col,row);
}*/

double Calibrations::getVDrift(int run, int det, int col, int row) const
{
  if (mChamberCalibrations && mLocalVDrift)
    return (double)mChamberCalibrations->getVDrift(det) * (double)mLocalVDrift->getValue(det, col, row);
  else
    return -1;
}

double Calibrations::getT0(int run, int det, int col, int row) const
{
  if (mChamberCalibrations && mLocalT0)
    return (double)mChamberCalibrations->getT0(det) * (double)mLocalT0->getValue(det, col, row);
  else
    return -1;
}
double Calibrations::getExB(int run, int det, int col, int row) const
{
  if (mChamberCalibrations)
    return (double)mChamberCalibrations->getExB(det);
  else
    return -1;
}
double Calibrations::getGainFactor(int run, int det, int col, int row) const
{
  if (mChamberCalibrations && mLocalGainFactor)
    return (double)mChamberCalibrations->getGainFactor(det) * (double)mLocalGainFactor->getValue(det, col, row);
  else
    return -1;
}
