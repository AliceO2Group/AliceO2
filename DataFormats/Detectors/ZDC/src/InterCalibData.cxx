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
#include "DataFormatsZDC/InterCalibData.h"

using namespace o2::zdc;

void InterCalibData::print() const
{
  const char* dn[NH] = {"ZNA", "ZPA", "ZNC", "ZPC", "ZEM"};
  for (int i = 0; i < NH; i++) {
    LOGF(info, "%s", dn[i]);
    for (int j = 0; j < NPAR; j++) {
      for (int k = 0; k < NPAR; k++) {
        if (k == 0) {
          printf("%e", mSum[i][j][k]);
        } else {
          printf(" %e", mSum[i][j][k]);
        }
      }
      printf("\n");
    }
  }
}
