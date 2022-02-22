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

#include <FairMQDevice.h>
#include "Headers/DataHeader.h"
#include "Framework/Logger.h"
#include "Framework/ProcessingContext.h"
#include "Framework/RawDeviceService.h"
#include "Framework/DataRefUtils.h"
#include "Framework/InputRecord.h"
#include "Framework/ServiceRegistry.h"
#include "CCDB/BasicCCDBManager.h"
#include "CommonConstants/LHCConstants.h"

#include "TPCWorkflow/ProcessingHelpers.h"

using namespace o2::framework;
using namespace o2::tpc;

// taken from CTFWriterSpec, TODO: should this be put to some more general location?
uint64_t processing_helpers::getRunNumber(ProcessingContext& pc)
{
  const std::string NAStr = "NA";

  uint64_t run = 0;
  const auto dh = DataRefUtils::getHeader<o2::header::DataHeader*>(pc.inputs().getFirstValid(true));
  if (dh->runNumber != 0) {
    run = dh->runNumber;
  }
  // check runNumber with FMQ property, if set, override DH number
  {
    auto runNStr = pc.services().get<RawDeviceService>().device()->fConfig->GetProperty<std::string>("runNumber", NAStr);
    if (runNStr != NAStr) {
      size_t nc = 0;
      auto runNProp = std::stol(runNStr, &nc);
      if (nc != runNStr.size()) {
        LOGP(error, "Property runNumber={} is provided but is not a number, ignoring", runNStr);
      } else {
        run = runNProp;
      }
    }
  }

  return run;
}

uint32_t processing_helpers::getCurrentTF(o2::framework::ProcessingContext& pc)
{
  return o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(pc.inputs().getFirstValid(true))->tfCounter;
}

uint64_t processing_helpers::getCreationTime(o2::framework::ProcessingContext& pc)
{
  return DataRefUtils::getHeader<DataProcessingHeader*>(pc.inputs().getFirstValid(true))->creation;
}

uint64_t processing_helpers::getTimeStamp(o2::framework::ProcessingContext& pc, o2::ccdb::BasicCCDBManager& ccdbManager)
{
  const auto tfOrbitFirst = DataRefUtils::getHeader<o2::header::DataHeader*>(pc.inputs().getFirstValid(true))->firstTForbit;
  const auto* tv = ccdbManager.getForTimeStamp<std::vector<Long64_t>>("CTP/Calib/OrbitReset", processing_helpers::getCreationTime(pc));
  const long tPrec = (*tv)[0] + tfOrbitFirst * o2::constants::lhc::LHCOrbitMUS; // microsecond-precise time stamp
  return tPrec;
}
