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

