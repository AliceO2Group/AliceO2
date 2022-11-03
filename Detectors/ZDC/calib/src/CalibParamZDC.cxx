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

#include "Framework/Logger.h"
#include "ZDCCalib/CalibParamZDC.h"

O2ParamImpl(o2::zdc::CalibParamZDC);

void o2::zdc::CalibParamZDC::print() const
{
  bool printed = false;
  if (rootOutput) {
    if (!printed) {
      LOG(info) << "CalibParamZDC::print()";
      printed = true;
    }
    printf("rootOutput=%s\n", rootOutput ? "true" : "false");
  }
  if (debugOutput) {
    if (!printed) {
      LOG(info) << "CalibParamZDC::print()";
      printed = true;
    }
    printf("debugOutput=%s\n", debugOutput ? "true" : "false");
  }
  if (outputDir.compare("./")) {
    if (!printed) {
      LOG(info) << "CalibParamZDC::print()";
      printed = true;
    }
    printf("outputDir=%s\n", outputDir.data());
  }
  if (metaFileDir.compare("/dev/null")) {
    if (!printed) {
      LOG(info) << "CalibParamZDC::print()";
      printed = true;
    }
    printf("metaFileDir=%s\n", metaFileDir.data());
  }
  if (descr.size() > 0) {
    if (!printed) {
      LOG(info) << "CalibParamZDC::print()";
      printed = true;
    }
    printf("descr=%s\n", descr.data());
  }
}
