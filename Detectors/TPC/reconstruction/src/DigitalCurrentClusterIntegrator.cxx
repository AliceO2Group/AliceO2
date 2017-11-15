// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitalCurrentClusterIntegrator.cxx
/// \author David Rohr

#include "TPCReconstruction/DigitalCurrentClusterIntegrator.h"

using namespace o2::TPC;

void DigitalCurrentClusterIntegrator::clear()
{
  for (int i = 0;i < Constants::MAXSECTOR;i++)
  {
    for (int j = 0;j < Constants::MAXGLOBALPADROW;j++)
    {
      if (mIntegratedCurrents[i][j])
      {
        int nPads = Mapper::instance().getNumberOfPadsInRowSector(j);
        memset(&mIntegratedCurrents[i][j][0], 0, nPads * sizeof(mIntegratedCurrents[i][j][0]));
      }
    }
  }
}

void DigitalCurrentClusterIntegrator::reset()
{
  for (int i = 0;i < Constants::MAXSECTOR;i++)
  {
    for (int j = 0;j < Constants::MAXGLOBALPADROW;j++)
    {
      mIntegratedCurrents[i][j].reset(nullptr);
    }
  }
}
