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

/// \class EMCALCalibrationHandler
/// \brief  Apply the bad channel, time and energy calibration
/// \author Joshua Koenig <joshua.konig@cern.ch>
/// \ingroup EMCALCalib
/// \since Aug 17, 2022

#ifndef EMCAL_CALIBRATOR_HANDLER_H_
#define EMCAL_CALIBRATOR_HANDLER_H_

#include "DataFormatsEMCAL/Cell.h"
#include "EMCALCalib/BadChannelMap.h"
#include "EMCALCalib/TimeCalibrationParams.h"
#include "EMCALCalib/EMCALCalibCCDBHelper.h"
#include "CCDB/BasicCCDBManager.h"

namespace o2
{

namespace emcal
{

template <class CellInputType>
class EMCALCalibrationHandler
{

 public:
  ///\brief constructor
  EMCALCalibrationHandler() = default;
  ///\brief destructor
  ~EMCALCalibrationHandler() = default;

  ///\brief check if cell is good cell
  ///\param towerID cell id to be checked
  ///\return true if cell is good or no calibration should be applied, false if cell is bad
  bool acceptCell(const int towerID);

  ///\brief calibrate cell time and energy
  ///\param cell input cell
  ///\return calibrated cell
  CellInputType getCellCalibrated(CellInputType cell);

  ClassDefNV(EMCALCalibrationHandler, 1);
};

template <class CellInputType>
bool EMCALCalibrationHandler<CellInputType>::acceptCell(const int towerID)
{
  if (!o2::emcal::EMCALCalibCCDBHelper::instance().isCalibrateBadChannels()) {
    LOG(debug) << "Bad Channels will not be calibrated";
    return true;
  }
  if ((o2::emcal::EMCALCalibCCDBHelper::instance().getBadChannelMap())->getChannelStatus(towerID) != o2::emcal::BadChannelMap::MaskType_t::GOOD_CELL) {
    return false;
  }
  return true;
}

template <class CellInputType>
CellInputType EMCALCalibrationHandler<CellInputType>::getCellCalibrated(CellInputType cell)
{
  const bool isLowGain = false; // for now, treat all cells as high gain cells as low gain calibration would need a lot of statistic that is not aggregated in the onlne system

  // get cell time calibration factor
  float timeShift = 0;
  if (o2::emcal::EMCALCalibCCDBHelper::instance().isCalibrateTime()) {
    timeShift = o2::emcal::EMCALCalibCCDBHelper::instance().getTimeCalibParams()->getTimeCalibParam(cell.getTower(), isLowGain);
  } else {
    LOG(debug) << "Cell time will not be calibrated";
  }

  // TODO: apply time slope calibration (have to push this to ccdb once verified)

  // get cell energy calibration factor
  float energyShift = 1;
  if (o2::emcal::EMCALCalibCCDBHelper::instance().isCalibrateGain()) {
    energyShift = o2::emcal::EMCALCalibCCDBHelper::instance().getGainCalibParams()->getGainCalibFactors(cell.getTower());
  } else {
    LOG(debug) << "Cell energy will not be calibrated";
  }

  // create calibrated cell
  float cellEnergyCalibrated = cell.getEnergy() * energyShift; // multiply be gain calib
  float cellTimeCalibrated = cell.getTimeStamp() - timeShift;  // shift timing signal
  CellInputType calibratedCell(cell.getTower(), cellEnergyCalibrated, cellTimeCalibrated, cell.getType());
  return calibratedCell;
}

} // namespace emcal
} // namespace o2

#endif