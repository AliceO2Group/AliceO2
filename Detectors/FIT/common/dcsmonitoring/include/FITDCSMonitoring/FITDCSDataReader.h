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

/// \file FITDCSDataReader.h
/// \brief DCS data point reader for FIT
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#ifndef O2_FIT_DCSDATAREADER_H
#define O2_FIT_DCSDATAREADER_H

#include "CCDB/CcdbObjectInfo.h"
#include "DataFormatsFIT/DCSDPValues.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "Rtypes.h"

#include <gsl/gsl>
#include <string>
#include <unordered_map>
#include <vector>
#include <unordered_map>

namespace o2
{
namespace fit
{

class FITDCSDataReader
{
 public:
  using DPID = o2::dcs::DataPointIdentifier;
  using DPVAL = o2::dcs::DataPointValue;
  using DPCOM = o2::dcs::DataPointCompositeObject;
  using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;

  FITDCSDataReader() = default;
  ~FITDCSDataReader() = default;

  void init(const std::vector<DPID>& pids);
  int process(const gsl::span<const DPCOM> dps);
  int processDP(const DPCOM& dpcom);
  uint64_t processFlags(uint64_t flag, const char* pid);
  void updateCcdbObjectInfo();

  const std::unordered_map<DPID, DCSDPValues>& getDpData() const;
  void resetDpData();
  const std::string& getCcdbPath() const;
  void setCcdbPath(const std::string& ccdbPath);
  long getStartValidity() const;
  void setStartValidity(long startValidity);
  bool isStartValiditySet() const;
  void resetStartValidity();
  long getEndValidity() const;
  const CcdbObjectInfo& getccdbDPsInfo() const;
  CcdbObjectInfo& getccdbDPsInfo();

  bool getVerboseMode() const;
  void setVerboseMode(bool verboseMode = true);

 private:
  std::unordered_map<DPID, o2::fit::DCSDPValues> mDpData; // the object that will go to the CCDB
  std::unordered_map<DPID, bool> mPids;                   // contains all PIDs for the processor, the bool
                                                          // will be true if the DP was processed at least once
  std::unordered_map<DPID, DPVAL> mDpsMap;                // this is the map that will hold the DPs

  std::string mCcdbPath;
  long mStartValidity = o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP; // TF index for processing, used to store CCDB object for DPs
  CcdbObjectInfo mCcdbDpInfo;

  union DPValueConverter {
    uint64_t raw_data;
    double double_value;
    uint uint_value;
  } dpValueConverter;

  bool mVerbose = false;

  ClassDefNV(FITDCSDataReader, 0);
}; // end class

} // namespace fit
} // namespace o2

#endif // O2_FIT_DCSDATAREADER_H
