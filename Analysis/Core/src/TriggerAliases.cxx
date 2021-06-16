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

#include "AnalysisCore/TriggerAliases.h"
#include "Framework/Logger.h"

void TriggerAliases::AddClassIdToAlias(uint32_t aliasId, int classId)
{
  if (classId < 0 || classId > 99) {
    LOGF(fatal, "Invalid classId = %d for aliasId = %d\n", classId, aliasId);
  } else if (classId < 50) {
    mAliasToTriggerMask[aliasId] |= 1ull << classId;
  } else {
    mAliasToTriggerMaskNext50[aliasId] |= 1ull << (classId - 50);
  }
}
