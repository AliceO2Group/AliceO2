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

#ifndef EMCAL_PEDESTAL_CALIB_DEVICE_H_
#define EMCAL_PEDESTAL_CALIB_DEVICE_H_

#include <memory>
#include <vector>

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "EMCALBase/Mapper.h"
#include "EMCALBase/Geometry.h"
#include "EMCALCalibration/PedestalProcessorData.h"
#include "EMCALCalibration/EMCALCalibExtractor.h"
#include "EMCALCalibration/EMCALPedestalHelper.h"

namespace o2::emcal
{

class PedestalCalibDevice : o2::framework::Task
{
 public:
  PedestalCalibDevice(bool dumpToFile, bool addRunNum) : mDumpToFile(dumpToFile), mAddRunNumber(addRunNum){};
  ~PedestalCalibDevice() final = default;

  void init(framework::InitContext& ctx) final;

  void run(framework::ProcessingContext& ctx) final;

  void sendData(o2::framework::EndOfStreamContext& ec, const Pedestal& data) const;

  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

  void resetStartTS() { mStartTS = o2::ccdb::getCurrentTimestamp(); }

  static const char* getPedDataBinding() { return "PEDData"; }

 private:
  Geometry* mGeometry = nullptr;                  ///< pointer to the emcal geometry class
  o2::emcal::EMCALCalibExtractor mCalibExtractor; ///< instance of the calibration extraction class                                                                ///< Calibration postprocessing
  PedestalProcessorData mPedestalData;            ///< pedestal data to accumulate data
  long int mStartTS = 0;                          ///< timestamp at the start of run used for the object in the ccdb
  bool mDumpToFile;                               ///< if output of pedestal calib (DCS ccdb) should be written to text file
  int mRun = 0;                                   ///< current run number
  bool mAddRunNumber = false;                     ///< if true, runNumber will be added to ccdb string
};

o2::framework::DataProcessorSpec getPedestalCalibDevice(bool dumpToFile, bool addRunNum);

} // end namespace o2::emcal

#endif