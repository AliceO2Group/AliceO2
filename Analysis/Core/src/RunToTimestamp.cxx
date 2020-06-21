// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Analysis/RunToTimestamp.h"

ClassImp(RunToTimestamp);

bool RunToTimestamp::insert(uint runNumber, long timestamp)
{
  std::pair<std::map<uint, long>::iterator, bool> check;
  check = mMap.insert(std::pair<uint, long>(runNumber, timestamp));
  if (!check.second) {
    LOG(FATAL) << "Run number " << runNumber << " already existed with a timestamp of " << check.first->second;
    return false;
  }
  LOG(INFO) << "Add new run " << runNumber << " with timestamp " << timestamp << " to converter";
  return true;
}

long RunToTimestamp::getTimestamp(uint run) const
{
  if (!Has(run)) {
    LOG(ERROR) << "Run to Timestamp converter does not have run " << run;
    return 0;
  }
  return mMap.at(run);
}

void RunToTimestamp::print() const
{
  LOG(INFO) << "Printing run number -> timestamp conversion";
  for (auto e : mMap) {
    LOG(INFO) << "Run number: " << e.first << " timestamp: " << e.second << "\n";
  }
}
