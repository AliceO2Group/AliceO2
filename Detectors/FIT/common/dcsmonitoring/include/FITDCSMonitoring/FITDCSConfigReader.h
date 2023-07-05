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

/// \file FITDCSConfigReader.h
/// \brief DCS configuration reader for FIT
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#ifndef O2_FIT_DCSCONFIGREADER_H
#define O2_FIT_DCSCONFIGREADER_H

#include "CCDB/CcdbObjectInfo.h"
#include "DataFormatsFIT/DeadChannelMap.h"

#include <gsl/span>
#include <string>

namespace o2
{
namespace fit
{

class FITDCSConfigReader
{
 public:
  FITDCSConfigReader() = default;
  ~FITDCSConfigReader() = default;

  virtual void processDChM(gsl::span<const char> configBuf);
  void updateDChMCcdbObjectInfo();

  const o2::fit::DeadChannelMap& getDChM() const;
  void resetDChM();
  const std::string& getCcdbPathDChm() const;
  void setCcdbPathDChM(const std::string& ccdbPath);
  const long getStartValidityDChM() const;
  const long getEndValidityDChM() const;
  void setStartValidityDChM(const long startValidity);
  const bool isStartValidityDChMSet() const;
  void resetStartValidityDChM();
  const o2::ccdb::CcdbObjectInfo& getObjectInfoDChM() const;
  o2::ccdb::CcdbObjectInfo& getObjectInfoDChM();

  const std::string& getFileNameDChM() const;
  void setFileNameDChM(const std::string& fileName);

  const uint getValidDaysDChM() const;
  void setValidDaysDChM(const uint validDays);

  const bool getVerboseMode() const;
  void setVerboseMode(const bool verboseMode);

  const bool getValidateUploadMode() const;
  void setValidateUploadMode(const bool validateUpload);

 protected:
  o2::fit::DeadChannelMap mDChM; ///< The dead channel map CCDB object
  bool mVerbose = false;         ///< Verbose mode

 private:
  std::string mFileNameDChM;                                              ///< The expected file name of the dead channel map
  uint mValidDaysDChM = 180u;                                             ///< The dead channel map validity in days
  std::string mCcdbPathDChM;                                              ///< The dead channel map CCDB path
  long mStartValidityDChM = o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP; ///< Start validity of the dead channel map CCDB object
  o2::ccdb::CcdbObjectInfo mCcdbObjectInfoDChM;                           ///< CCDB object info for the dead channel map
  bool mValidateUpload = true;                                            ///< Validate upload mode

  ClassDefNV(FITDCSConfigReader, 1);
};

} // namespace fit
} // namespace o2

#endif // O2_FIT_DCSCONFIGREADER_H