  /// \file Clusterer.cxx
/// \brief Base class for TPC Clusterer


#include "TPCSimulation/Clusterer.h"
#include "TPCSimulation/ClusterContainer.h"

ClassImp(AliceO2::TPC::Clusterer)

using namespace AliceO2::TPC;

//________________________________________________________________________
Clusterer::Clusterer(
    Int_t rowsMax,
    Int_t padsMax,
    Int_t timeBinsMax,
    Int_t minQMax,
    Bool_t requirePositiveCharge,
    Bool_t requireNeighbouringPad):
  TObject(),
  mClusterContainer(nullptr),
  mRowsMax(rowsMax),
  mPadsMax(padsMax),
  mTimeBinsMax(timeBinsMax),
  mMinQMax(minQMax),
  mRequirePositiveCharge(requirePositiveCharge),
  mRequireNeighbouringPad(requireNeighbouringPad)
{
}

//________________________________________________________________________
Clusterer::~Clusterer()
{
}

