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

/// \file FITDCSConfigReader.cxx
/// \brief FIT reader for DCS configurations
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#include "FITDCSMonitoring/FITDCSConfigReader.h"

#include "DetectorsCalibration/Utils.h"

#include <cstdint>
#include <sstream>

using namespace o2::fit;

void FITDCSConfigReader::processBChM(gsl::span<const char> configBuf)
{
  LOG(info) << "Processing bad channel map";

  // AM: need to specify the size,
  // otherwise the configBuf.data() pointer might point to an array that is too long,
  // the array sometimes includes "extra memory from previous runs"
  std::istringstream iss(std::string(configBuf.data(), configBuf.size()));
  uint8_t iLine = 0; // line 0 corresponds to cahnnel id 0 and so on

  for (std::string line; std::getline(iss, line);) {
    mBChM.setChannelGood(iLine, std::stoi(line) == 1);
    iLine++;
  }

  if (getVerboseMode()) {
    LOGP(info, "Processed {} channels", iLine);
  }
}

void FITDCSConfigReader::updateBChMCcdbObjectInfo()
{
  std::map<std::string, std::string> metadata;
  o2::calibration::Utils::prepareCCDBobjectInfo(mBChM, mCcdbObjectInfoBChM, mCcdbPathBChM, metadata, getStartValidityBChM(), getEndValidityBChM());
}

const o2::fit::BadChannelMap& FITDCSConfigReader::getBChM() const { return mBChM; }
void FITDCSConfigReader::resetBChM() { mBChM.clear(); }
const std::string& FITDCSConfigReader::getCcdbPathBChm() const { return mCcdbPathBChM; }
void FITDCSConfigReader::setCcdbPathBChM(const std::string& ccdbPath) { mCcdbPathBChM = ccdbPath; }
const long FITDCSConfigReader::getStartValidityBChM() const { return mStartValidityBChM; }
const long FITDCSConfigReader::getEndValidityBChM() const { return mStartValidityBChM + o2::ccdb::CcdbObjectInfo::MONTH; }
void FITDCSConfigReader::setStartValidityBChM(const long startValidity) { mStartValidityBChM = startValidity; }
const bool FITDCSConfigReader::isStartValidityBChMSet() const { return mStartValidityBChM != o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP; }
void FITDCSConfigReader::resetStartValidityBChM() { mStartValidityBChM = o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP; }
const o2::ccdb::CcdbObjectInfo& FITDCSConfigReader::getObjectInfoBChM() const { return mCcdbObjectInfoBChM; }
o2::ccdb::CcdbObjectInfo& FITDCSConfigReader::getObjectInfoBChM() { return mCcdbObjectInfoBChM; }

const std::string& FITDCSConfigReader::getFileNameBChM() const { return mFileNameBChM; }
void FITDCSConfigReader::setFileNameBChM(const std::string& fileName) { mFileNameBChM = fileName; }

const bool FITDCSConfigReader::getVerboseMode() const { return mVerbose; }
void FITDCSConfigReader::setVerboseMode(const bool verboseMode) { mVerbose = verboseMode; }