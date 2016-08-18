/// \file Cluster.cxx
/// \brief Cluster structure for TPC clusters

#include "TPCsimulation/Cluster.h"

ClassImp(AliceO2::TPC::Cluster)

using namespace AliceO2::TPC;

//________________________________________________________________________
Cluster::Cluster():
mCRU(-1),
mRow(-1),
mQ(-1),
mQmax(-1),
mPadMean(-1),
mPadSigma(-1),
mTimeMean(-1),
mTimeSigma(-1),
FairTimeStamp()
{
}

//________________________________________________________________________
Cluster::Cluster(Short_t cru, Short_t row, Float_t q, Float_t qmax,
		 Float_t padmean, Float_t padsigma,
		 Float_t timemean, Float_t timesigma):
mCRU(cru),
mRow(row),
mQ(q),
mQmax(qmax),
mPadMean(padmean),
mPadSigma(padsigma),
mTimeMean(timemean),
mTimeSigma(timesigma),
FairTimeStamp()
{
}

//________________________________________________________________________
Cluster::~Cluster()
{
}

//________________________________________________________________________
void Cluster::setParameters(Short_t cru, Short_t row, Float_t q, Float_t qmax,
			    Float_t padmean, Float_t padsigma,
			    Float_t timemean, Float_t timesigma)
{
  mCRU = cru;
  mRow = row;
  mQ = q;
  mQmax = qmax;
  mPadMean = padmean;
  mPadSigma = padsigma;
  mTimeMean = timemean;
  mTimeSigma = timesigma;
}


//________________________________________________________________________
std::ostream &Cluster::Print(std::ostream &output) const
{
  output << "TPC Cluster in CRU [" << mCRU << "], pad row ["
	 << mRow << "] with charge/maxCharge " << mQ << "/" << mQmax
	 << "and coordinates (" << mPadMean << ", " << mTimeMean << ")"
	 << "and width (" << mPadSigma << ", " << mTimeSigma << ")";
  return output;
}
