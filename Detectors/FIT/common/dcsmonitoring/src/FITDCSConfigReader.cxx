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

void FITDCSConfigReader::processDChM(gsl::span<const char> configBuf)
{
  LOG(info) << "Processing dead channel map";

  // AM: need to specify the size,
  // otherwise the configBuf.data() pointer might point to an array that is too long,
  // the array sometimes includes "extra memory from previous runs"
  std::istringstream iss(std::string(configBuf.data(), configBuf.size()));
  uint8_t iLine = 0; // line 0 corresponds to cahnnel id 0 and so on

  for (std::string line; std::getline(iss, line);) {
    mDChM.setChannelAlive(iLine, std::stoi(line) == 1);
    iLine++;
  }

  if (getVerboseMode()) {
    LOGP(info, "Processed {} channels", iLine);
  }
}

void FITDCSConfigReader::updateDChMCcdbObjectInfo()
{
  std::map<std::string, std::string> metadata;
  o2::calibration::Utils::prepareCCDBobjectInfo(mDChM, mCcdbObjectInfoDChM, mCcdbPathDChM, metadata, getStartValidityDChM(), getEndValidityDChM());
  mCcdbObjectInfoDChM.setValidateUpload(getValidateUploadMode());
}

const o2::fit::DeadChannelMap& FITDCSConfigReader::getDChM() const { return mDChM; }
void FITDCSConfigReader::resetDChM() { mDChM.clear(); }
const std::string& FITDCSConfigReader::getCcdbPathDChm() const { return mCcdbPathDChM; }
void FITDCSConfigReader::setCcdbPathDChM(const std::string& ccdbPath) { mCcdbPathDChM = ccdbPath; }
const long FITDCSConfigReader::getStartValidityDChM() const { return mStartValidityDChM; }
const long FITDCSConfigReader::getEndValidityDChM() const { return mStartValidityDChM + getValidDaysDChM() * o2::ccdb::CcdbObjectInfo::DAY; }
void FITDCSConfigReader::setStartValidityDChM(const long startValidity) { mStartValidityDChM = startValidity; }
const bool FITDCSConfigReader::isStartValidityDChMSet() const { return mStartValidityDChM != o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP; }
void FITDCSConfigReader::resetStartValidityDChM() { mStartValidityDChM = o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP; }
const o2::ccdb::CcdbObjectInfo& FITDCSConfigReader::getObjectInfoDChM() const { return mCcdbObjectInfoDChM; }
o2::ccdb::CcdbObjectInfo& FITDCSConfigReader::getObjectInfoDChM() { return mCcdbObjectInfoDChM; }

const std::string& FITDCSConfigReader::getFileNameDChM() const { return mFileNameDChM; }
void FITDCSConfigReader::setFileNameDChM(const std::string& fileName) { mFileNameDChM = fileName; }

const uint FITDCSConfigReader::getValidDaysDChM() const { return mValidDaysDChM; }
void FITDCSConfigReader::setValidDaysDChM(const uint validDays) { mValidDaysDChM = validDays; }

const bool FITDCSConfigReader::getVerboseMode() const { return mVerbose; }
void FITDCSConfigReader::setVerboseMode(const bool verboseMode) { mVerbose = verboseMode; }

const bool FITDCSConfigReader::getValidateUploadMode() const { return mValidateUpload; }
void FITDCSConfigReader::setValidateUploadMode(const bool validateUpload) { mValidateUpload = validateUpload; };