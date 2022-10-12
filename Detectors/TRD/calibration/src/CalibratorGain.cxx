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

/// \file CalibratorGain.cxx
/// \brief TimeSlot-based calibration of gain
/// \author Felix Schlepper

#include "TRDCalibration/CalibratorGain.h"
#include "TStopwatch.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include <string>
#include <map>
#include <memory>
#include "CommonUtils/NameConf.h"
#include "CommonUtils/MemFileHelper.h"

using namespace o2::trd::constants;

namespace o2::trd
{

using Slot = o2::calibration::TimeSlot<GainCalibration>;

void CalibratorGain::initOutput()
{
  // reset the CCDB output vectors
  mInfoVector.clear();
  mObjectVector.clear();
}

void CalibratorGain::initProcessing()
{
  if (mInitDone) {
    return;
  }
}

void CalibratorGain::finalizeSlot(Slot& slot)
{
  // do actual calibration for the data provided in the given slot
  TStopwatch timer;
  timer.Start();
  // TODO processing
  timer.Stop();
  LOGF(info, "Done fitting angular residual histograms. CPU time: %f, real time: %f", timer.CpuTime(), timer.RealTime());

  // Write results to file
  if (mEnableOutput) {
  }

  // assemble CCDB object
  CalGain calObject;
  auto clName = o2::utils::MemFileHelper::getClassName(calObject);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  std::map<std::string, std::string> metadata; // TODO: do we want to store any meta data?
  long startValidity = slot.getStartTimeMS() - 10 * o2::ccdb::CcdbObjectInfo::SECOND;
  mInfoVector.emplace_back("TRD/Calib/Gain", clName, flName, metadata, startValidity, startValidity + o2::ccdb::CcdbObjectInfo::HOUR);
  mObjectVector.push_back(calObject);
}

Slot& CalibratorGain::emplaceNewSlot(bool front, TFType tStart, TFType tEnd)
{
  auto& container = getSlots();
  auto& slot = front ? container.emplace_front(tStart, tEnd) : container.emplace_back(tStart, tEnd);
  slot.setContainer(std::make_unique<GainCalibration>());
  return slot;
}

} // namespace o2::trd
