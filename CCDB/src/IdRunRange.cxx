// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//  defines the run validity range of the object:                  //
//  [mFirstRun, mLastRun]                                          //
#include "CCDB/IdRunRange.h"
#include <FairLogger.h>  // for LOG

using namespace o2::CDB;

ClassImp(IdRunRange)

IdRunRange::IdRunRange() : mFirstRun(-1), mLastRun(-1)
{
  // constructor
}

IdRunRange::IdRunRange(Int_t firstRun, Int_t lastRun) : mFirstRun(firstRun), mLastRun(lastRun)
{
  // constructor
}

IdRunRange::~IdRunRange()
{
  // destructor
}

Bool_t IdRunRange::isOverlappingWith(const IdRunRange &other) const
{
  // check if this runRange overlaps other runRange

  if (!(isValid() && other.isValid())) {
    LOG(ERROR) << "Comparing invalid run ranges!" << FairLogger::endl;
    return kFALSE;
  }

  if (isAnyRange() || other.isAnyRange()) {
    LOG(ERROR) << "Comparing unspecified ranges!" << FairLogger::endl;
    return kFALSE;
  }

  return ((mFirstRun < other.mFirstRun && other.mFirstRun <= mLastRun) ||
          (other.mFirstRun <= mFirstRun && mFirstRun <= other.mLastRun));
}

Bool_t IdRunRange::isSupersetOf(const IdRunRange &other) const
{
  // check if this runRange contains other runRange

  if (!(isValid() && other.isValid())) {
    LOG(ERROR) << "Comparing invalid run ranges!" << FairLogger::endl;
    return kFALSE;
  }

  if (isAnyRange()) {
    return kTRUE;
  }

  return mFirstRun <= other.mFirstRun && other.mFirstRun <= mLastRun && mFirstRun <= other.mLastRun &&
         other.mLastRun <= mLastRun;
}

Bool_t IdRunRange::isEqual(const TObject *obj) const
{
  // check if this runRange is equal to other runRange

  if (this == obj) {
    return kTRUE;
  }

  if (IdRunRange::Class() != obj->IsA()) {
    return kFALSE;
  }
  IdRunRange *other = (IdRunRange *) obj;
  return mFirstRun == other->mFirstRun && mLastRun == other->mLastRun;
}

Bool_t IdRunRange::isValid() const
{
  // validity check

  if (mFirstRun < 0 && mLastRun < 0) {
    return kTRUE;
  }

  if (mFirstRun >= 0 && mLastRun >= mFirstRun) {
    return kTRUE;
  }

  return kFALSE;
}
