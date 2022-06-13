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

#include "DetectorsDCS/RunStatusChecker.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"

using namespace o2::dcs;

const o2::parameters::GRPECSObject* RunStatusChecker::check(long ts)
{
  if (ts < 0) {
    ts = o2::ccdb::getCurrentTimestamp();
  }
  if (ts <= mLastTimeStampChecked) {
    LOGP(alarm, "RunStatusChecker::check was called with decreasing timestamp {}, previous was {}", ts, mLastTimeStampChecked);
    return nullptr;
  }
  mLastTimeStampChecked = ts;

  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  bool fatalOn = mgr.getFatalWhenNull();
  mgr.setFatalWhenNull(false);
  if (mRunStatus == RunStatus::STOP) { // the STOP was detected at previous check
    mRunFollowed = -1;
    mRunStatus = RunStatus::NONE;
  }
  std::map<std::string, std::string> md;
  if (mRunFollowed > 0) { // run start was seen
    md["runNumber"] = std::to_string(mRunFollowed);
  }
  const auto* grp = mgr.getSpecific<o2::parameters::GRPECSObject>("GLO/Config/GRPECS", ts, md);
  if (grp) { // some object was returned
    if (mRunFollowed > 0) {
      if ((ts > grp->getTimeEnd()) && (grp->getTimeEnd() > grp->getTimeStart())) { // this means that the EOR was registered
        mRunStatus = RunStatus::STOP;
      } else { // run still continues
        mRunStatus = RunStatus::ONGOING;
      }
    } else {                                                // we were not following detector run, check if the current one has asked detectors
      if ((grp->getDetsReadOut() & mDetMask) == mDetMask) { // we start following this run
        if (grp->getTimeEnd() > grp->getTimeStart()) {
          if (ts < grp->getTimeEnd()) { // only in tests with ad hoc ts the ts_EOR can be seen > ts
            mRunStatus = RunStatus::START;
            mRunFollowed = grp->getRun();
          }
        } else {
          mRunStatus = RunStatus::START;
          mRunFollowed = grp->getRun();
        }
      }
    }
  } else {                  // query did not return any GRP -> we are certainly not in the wanted detectors run
    if (mRunFollowed > 0) { // normally this should not happen
      LOGP(warning, "We were following {} run {} but the query at {} did not return any GRP, problem with EOR?", o2::detectors::DetID::getNames(mDetMask), mRunFollowed, ts);
      mRunStatus = RunStatus::STOP;
    }
  }
  mgr.setFatalWhenNull(fatalOn);
  return mRunStatus == RunStatus::NONE ? nullptr : grp;
}
