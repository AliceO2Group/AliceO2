// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

  /// \file Clusterer.cxx
/// \brief Base class for TPC Clusterer


#include "TPCSimulation/Clusterer.h"


using namespace o2::TPC;

Clusterer::Clusterer()
  : Clusterer(18, 138, 1024, 5, true, true)
{
}

//________________________________________________________________________
Clusterer::Clusterer(int rowsMax, int padsMax, int timeBinsMax, int minQMax,
    bool requirePositiveCharge, bool requireNeighbouringPad)
  : mClusterContainer(nullptr)
  , mRowsMax(rowsMax)
  , mPadsMax(padsMax)
  , mTimeBinsMax(timeBinsMax)
  , mMinQMax(minQMax)
  , mRequirePositiveCharge(requirePositiveCharge)
  , mRequireNeighbouringPad(requireNeighbouringPad)
{
}

