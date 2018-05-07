// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/WorkflowSpec.h"
#include "Framework/runDataProcessing.h"

#include "SimReaderSpec.h"
#include "CollisionTimePrinter.h"

// for TPC
#include "TPCDriftTimeDigitizerSpec.h"

using namespace o2::framework;

/// This function is required to be implemented to define the workflow
/// specifications
void defineDataProcessing(WorkflowSpec& specs)
{
  specs.clear();

  int fanoutsize = 0;

  //
  // specs.emplace_back(o2::steer::getCollisionTimePrinter(fanoutsize++));

  // parallely processing the first 6 TPC sectors
  for (int s = 0; s < 1; ++s) {
    // probably a parallel construct can be used here
    specs.emplace_back(o2::steer::getTPCDriftTimeDigitizer(s, fanoutsize++));
  }

  specs.emplace_back(o2::steer::getSimReaderSpec(fanoutsize));
}
