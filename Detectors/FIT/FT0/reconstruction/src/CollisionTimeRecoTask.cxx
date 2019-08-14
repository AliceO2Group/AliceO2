// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  CollisionTimeRecoTask.cxx
/// \brief Implementation of the FIT reconstruction task

#include "FT0Reconstruction/CollisionTimeRecoTask.h"
#include "FairLogger.h" // for LOG

using namespace o2::ft0;

/*
//_____________________________________________________________________
CollisionTimeRecoTask::CollisionTimeRecoTask()
{
  // at the moment nothing.
}

*/

//_____________________________________________________________________
void CollisionTimeRecoTask::Process(const Digit& digits,
                                    RecPoints& recPoints) const
{
  //  LOG(INFO) << "Running reconstruction on new event";
  recPoints.fillFromDigits(digits);
}
//________________________________________________________
void CollisionTimeRecoTask::FinishTask()
{
  // finalize digitization, if needed, flash remaining digits
  // if (!mContinuous)   return;
}
