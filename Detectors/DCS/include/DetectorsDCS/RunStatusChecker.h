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

#ifndef O2_RUN_STATUS_CHECKER_H
#define O2_RUN_STATUS_CHECKER_H

/// @author ruben.shahoyan@cern.ch

#include "DataFormatsParameters/GRPECSObject.h"
#include "DetectorsCommonDataFormats/DetID.h"

/*
  Utility class to check the status of the run with particular detectors participating
  Usage: first create an instance for the mask of detectors which must be in the run, e.g.

  RunStatusChecker runChecker{ o2::detectors::DetID::getMask("EMC,PHS") };
  const o2::parameters::GRPECSObject* myGRPECS = nullptr;

  Then, periodically check:

  myGRPECS = runChecker.check();

  The check will set the status of the run with selected detectors (can be inspected by getRunStatus() method).

  if (check.getRunStatus() == o2::dcs::RunStatusChecker::RunStatus::NONE) {
    LOGP(info, "No run with {} is ongoing or finished", o2::detectors::DetID::getNames(checker->getDetectorsMask()) );
  }
  else if (check.getRunStatus() == o2::dcs::RunStatusChecker::RunStatus::START) { // saw new run with wanted detectors
    LOGP(info, "Run {} with {} has started", checker.getFollowedRun(), o2::detectors::DetID::getNames(checker->getDetectorsMask()) );
  }
  else if (check.getRunStatus() == o2::dcs::RunStatusChecker::RunStatus::ONGOING) { // run which was already seen is still ongoing
    LOGP(info, "Run {} with {} is still ongoing", checker.getFollowedRun(), o2::detectors::DetID::getNames(checker->getDetectorsMask()) );
  }
  else if (check.getRunStatus() == o2::dcs::RunStatusChecker::RunStatus::STOP) { // run which was already seen was stopped (EOR seen)
    LOGP(info, "Run {} with {} was stopped", checker.getFollowedRun(), o2::detectors::DetID::getNames(checker->getDetectorsMask()) );
  }

  In all cases except RunStatusChecker::NONE a const non-null pointer on the GRP of the followed run will be returned.

  By default the check will be done for the current timestamp, for test purposes one can call it with arbitrary increasing timestamps
*/

namespace o2::dcs
{

class RunStatusChecker
{
 public:
  enum class RunStatus { NONE,    // check did not find onging run with current detector
                         START,   // check found a new run started
                         ONGOING, // check found ongoing run which was already checked
                         STOP     // check found that previously ongoing run was stopped
  };

  RunStatusChecker() = delete;
  RunStatusChecker(o2::detectors::DetID::mask_t detmask) : mDetMask(detmask) {}

  RunStatus getRunStatus() const { return mRunStatus; }
  int getFollowedRun() const { return mRunFollowed; }
  o2::detectors::DetID::mask_t getDetectorsMask() const { return mDetMask; }
  const o2::parameters::GRPECSObject* check(long ts = -1);

 private:
  RunStatus mRunStatus = RunStatus::NONE;
  o2::detectors::DetID::mask_t mDetMask{};
  int mRunFollowed = -1; // particular run followed, assumption is that at the given moment there might be only run with particular detector
  long mLastTimeStampChecked = -1;

  ClassDefNV(RunStatusChecker, 0);
};

} // namespace o2::dcs

#endif
