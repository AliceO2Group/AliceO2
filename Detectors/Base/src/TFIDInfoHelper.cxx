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

#include "Framework/ProcessingContext.h"
#include "Framework/InputRecord.h"
#include "Framework/DataRefUtils.h"
#include "Framework/DataProcessingHeader.h"
#include "Headers/DataHeader.h"
#include "DetectorsBase/TFIDInfoHelper.h"

using namespace o2::framework;

void o2::base::TFIDInfoHelper::fillTFIDInfo(ProcessingContext& pc, o2::dataformats::TFIDInfo& ti)
{
  const auto ref = pc.inputs().getFirstValid(true);
  const auto* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
  const auto* dph = DataRefUtils::getHeader<DataProcessingHeader*>(ref);
  static int errCount = 0;
  if (dh->firstTForbit == -1U || dph->creation == -1) {
    if (errCount++ < 5) {
      LOGP(warn, "Ignoring gummy input with orbit {} and creation time {} in fillTFIDInfo", dh->firstTForbit, dph->creation);
    }
    return;
  }
  ti.firstTForbit = dh->firstTForbit;
  ti.tfCounter = dh->tfCounter;
  ti.runNumber = dh->runNumber;
  ti.startTime = dph->startTime;
  ti.creation = dph->creation;
}
