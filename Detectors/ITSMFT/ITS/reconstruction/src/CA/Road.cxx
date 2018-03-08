// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file Road.cxx
/// \brief
///

#include "ITSReconstruction/CA/Road.h"

namespace o2
{
namespace ITS
{
namespace CA
{

Road::Road() : mCellIds{}, mRoadSize{}, mIsFakeRoad{} { resetRoad(); }

Road::Road(int cellLayer, int cellId) : Road() { addCell(cellLayer, cellId); }

void Road::resetRoad()
{
  mCellIds.fill(Constants::ITS::UnusedIndex);
  mRoadSize = 0;
}

void Road::addCell(int cellLayer, int cellId)
{
  if (mCellIds[cellLayer] == Constants::ITS::UnusedIndex) {

    ++mRoadSize;
  }

  mCellIds[cellLayer] = cellId;
}
}
}
}
