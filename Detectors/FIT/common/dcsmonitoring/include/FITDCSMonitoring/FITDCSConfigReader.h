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
#include "DataFormatsFIT/BadChannelMap.h"

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

  virtual void processBChM(gsl::span<const char> configBuf);
  void updateBChMCcdbObjectInfo();

  const o2::fit::BadChannelMap& getBChM() const;
  void resetBChM();
  const std::string& getCcdbPathBChm() const;
  void setCcdbPathBChM(const std::string& ccdbPath);
  const long getStartValidityBChM() const;
  const long getEndValidityBChM() const;
  void setStartValidityBChM(const long startValidity);
  const bool isStartValidityBChMSet() const;
  void resetStartValidityBChM();
  const o2::ccdb::CcdbObjectInfo& getObjectInfoBChM() const;
  o2::ccdb::CcdbObjectInfo& getObjectInfoBChM();

  const std::string& getFileNameBChM() const;
  void setFileNameBChM(const std::string& fileName);

  const bool getVerboseMode() const;
  void setVerboseMode(const bool verboseMode);

 protected:
  o2::fit::BadChannelMap mBChM; ///< The bad channel map CCDB object
  bool mVerbose = false;        ///< Verbose mode

 private:
  std::string mFileNameBChM;                                              ///< The expected file name of the bad channel map
  std::string mCcdbPathBChM;                                              ///< The bad channel map CCDB path
  long mStartValidityBChM = o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP; ///< Start validity of the bad channel map CCDB object
  o2::ccdb::CcdbObjectInfo mCcdbObjectInfoBChM;                           ///< CCDB object info for the bad channel map

  ClassDefNV(FITDCSConfigReader, 1);
};

} // namespace fit
} // namespace o2

#endif // O2_FIT_DCSCONFIGREADER_H