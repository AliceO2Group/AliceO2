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

/// \file DigitalCurrentClusterIntegrator.cxx
/// \author David Rohr

#include "TPCReconstruction/DigitalCurrentClusterIntegrator.h"

using namespace o2::tpc;
using namespace o2::tpc::constants;

void DigitalCurrentClusterIntegrator::clear()
{
  for (int i = 0; i < MAXSECTOR; i++) {
    for (int j = 0; j < MAXGLOBALPADROW; j++) {
      if (mIntegratedCurrents[i][j]) {
        int nPads = Mapper::instance().getNumberOfPadsInRowSector(j);
        memset(&mIntegratedCurrents[i][j][0], 0, nPads * sizeof(mIntegratedCurrents[i][j][0]));
      }
    }
  }
}

void DigitalCurrentClusterIntegrator::reset()
{
  for (int i = 0; i < MAXSECTOR; i++) {
    for (int j = 0; j < MAXGLOBALPADROW; j++) {
      mIntegratedCurrents[i][j].reset(nullptr);
    }
  }
}
