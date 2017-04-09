/// \file BoxCluster.cxx
/// \brief Class to have some more info about the BoxClusterer clusters

#include "TPCSimulation/BoxCluster.h"
#include "TPCSimulation/Cluster.h"

ClassImp(o2::TPC::BoxCluster)

using namespace o2::TPC;

//________________________________________________________________________
BoxCluster::BoxCluster():
  Cluster(),
  mPad(-1),
  mTime(-1),
  mSize(-1)
{
}

//________________________________________________________________________
BoxCluster::BoxCluster(Short_t cru, Short_t row, Short_t pad, Short_t time,
		       Float_t q, Float_t qmax, Float_t padmean,
		       Float_t padsigma, Float_t timemean,
		       Float_t timesigma, Short_t size):
  Cluster(cru, row, q, qmax, padmean, padsigma, timemean, timesigma),
  mPad(pad),
  mTime(time),
  mSize(size)
{
}

//________________________________________________________________________
BoxCluster::~BoxCluster()
= default;

//________________________________________________________________________
void BoxCluster::setBoxParameters(Short_t pad, Short_t time, Short_t size)
{
  mPad = pad;
  mTime = time;
  mSize = size;
}

//________________________________________________________________________
std::ostream& BoxCluster::Print(std::ostream &output) const
{
  Cluster::Print(output);
  output << " centered at (pad, time) = " << mPad << ", " << mTime
	 << " covering " << Int_t(mSize/10)  << " pads and " << mSize%10
	 << " time bins";
  return output;
}
