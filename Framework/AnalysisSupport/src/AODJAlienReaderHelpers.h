// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

namespace o2::framework::readers
{

struct AODJAlienReaderHelpers {
  static AlgorithmSpec rootFileReaderCallback();
  static void dumpFileMetrics(o2::monitoring::Monitoring& monitoring, TFile* currentFile, uint64_t startedAt, int tfPerFile, int tfRead);
};

} // namespace o2::framework::readers

#endif // O2_FRAMEWORK_AODREADERHELPERS_H_
