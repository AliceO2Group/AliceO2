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

#include <string>
#include <fmt/format.h>
#include "Framework/Logger.h"
#include "ZDCCalib/CalibParamZDC.h"

O2ParamImpl(o2::zdc::CalibParamZDC);

int o2::zdc::CalibParamZDC::updateCcdbObjectInfo(o2::ccdb::CcdbObjectInfo& info) const
{
  // Tune the validity of calibration object
  auto eov = info.getEndValidityTimestamp();
  auto sov = info.getStartValidityTimestamp();
  LOG(info) << "Initial validity of calibration object: " << sov << ":" << eov;
  if (eovTune > 0) { // Absolute
    info.setEndValidityTimestamp(eovTune);
    if (eovTune < eov) {
      LOG(warning) << __func__ << " New EOV: " << eovTune << " < old EOV " << eov;
      return 1;
    } else {
      LOG(info) << __func__ << " Updating EOV from " << eov << " to " << eovTune;
      return 0;
    }
  } else if (eovTune < 0) { // Increase eov by -eovTune
    auto neov = eov - eovTune;
    info.setEndValidityTimestamp(neov);
    if (neov < eov) {
      // Should never happen unless there is an overflow
      LOG(error) << __func__ << " New EOV: " << neov << " < old EOV " << eov;
      return 1;
    } else {
      LOG(info) << __func__ << " Updating EOV from " << eov << " to " << neov;
      return 0;
    }
  }
  return 0;
}

void o2::zdc::CalibParamZDC::print() const
{
  std::string msg = "";
  bool printed = false;
  if (rootOutput) {
    msg = msg + fmt::format(" rootOutput={}", rootOutput ? "true" : "false");
  }
  if (debugOutput) {
    msg = msg + fmt::format(" debugOutput={}", debugOutput ? "true" : "false");
  }
  if (outputDir.compare("./")) {
    msg = msg + fmt::format(" outputDir={}", outputDir);
  }
  if (metaFileDir.compare("/dev/null")) {
    msg = msg + fmt::format(" metaFileDir={}", metaFileDir);
  }
  if (descr.size() > 0) {
    msg = msg + fmt::format(" descr={}", descr);
  }
  if (modTF > 0) {
    msg = msg + fmt::format(" modTF={}", modTF);
  }
  if (mCTimeMod > 0) {
    msg = msg + fmt::format(" mCTimeMod={}", mCTimeMod);
  }
  if (msg.size() > 0) {
    LOG(info) << "CalibParamZDC::print():" << msg;
  }
}
