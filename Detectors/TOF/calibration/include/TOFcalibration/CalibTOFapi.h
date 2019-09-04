// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CalibTOFapi.h
/// \brief Class to use TOF calibration (decalibration, calibration)
/// \author Francesco.Noferini@cern.ch, Chiara.Zampolli@cern.ch

#ifndef ALICEO2_GLOBTRACKING_CALIBTOFAPI_
#define ALICEO2_GLOBTRACKING_CALIBTOFAPI_

#include <iostream>
#include "CCDB/BasicCCDBManager.h"

namespace o2
{
  
namespace tof

class CalibTOFapi
{
 public:
  CalibTOFapi() = default;
  CalibTOFapi(const std::string db);
  ~CalibTOFapi = default;
  void setDB(const std::string db) {
    mCCDBserver = db;
  }
  
 private:

  std::string mCCDB;      ///< CCDB server where to look for the TOF CCDB objects
  
  ClassDefNV(CalibTOFapi, 1);
};
} // namespace tof
} // namespace o2
#endif
