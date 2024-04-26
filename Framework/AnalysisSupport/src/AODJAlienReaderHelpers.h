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

#ifndef O2_FRAMEWORK_AODJALIENREADERHELPERS_H_
#define O2_FRAMEWORK_AODJALIENREADERHELPERS_H_

#include "Framework/TableBuilder.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/Logger.h"
#include <Monitoring/Monitoring.h>

#include <uv.h>
class TFile;

namespace o2::framework::readers
{

struct AODJAlienReaderHelpers {
  static AlgorithmSpec rootFileReaderCallback();
  static void dumpFileMetrics(o2::monitoring::Monitoring& monitoring, TFile* currentFile, uint64_t startedAt, uint64_t ioTime, int tfPerFile, int tfRead);
};

} // namespace o2::framework::readers

#endif // O2_FRAMEWORK_AODREADERHELPERS_H_
