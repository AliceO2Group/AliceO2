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

#include "TOFCalibration/TOFDiagnosticCalibrator.h"
#include "Framework/Logger.h"
#include "DetectorsCalibration/Utils.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/CcdbApi.h"

namespace o2
{
namespace tof
{

using Slot = o2::calibration::TimeSlot<o2::tof::Diagnostic>;

//----------------------------------------------------------
void TOFDiagnosticCalibrator::initOutput()
{
  mccdbInfoVector.clear();
  mDiagnosticVector.clear();
}

//----------------------------------------------------------
void TOFDiagnosticCalibrator::finalizeSlot(Slot& slot)
{
  Diagnostic* diag = slot.getContainer();
  LOG(info) << "Finalizing slot";
  diag->print();
  std::map<std::string, std::string> md;
  if (mRunNumber > -1) {
    md["runNumber"] = std::to_string(mRunNumber);
  }

  auto clName = o2::utils::MemFileHelper::getClassName(*diag);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);

  uint64_t startingMS = slot.getStartTimeMS() - 10000; // start 10 seconds before
  uint64_t stoppingMS = slot.getEndTimeMS() + 10000;   // stop 10 seconds after
  mccdbInfoVector.emplace_back("TOF/Calib/Diagnostic", clName, flName, md, startingMS, stoppingMS);
  mDiagnosticVector.emplace_back(*diag);
}

//----------------------------------------------------------
Slot& TOFDiagnosticCalibrator::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{
  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<Diagnostic>());
  return slot;
}

} // end namespace tof
} // end namespace o2
