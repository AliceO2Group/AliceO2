/// \file Cluster.cxx
/// \brief Cluster structure for TPC clusters

#include "TPCSimulation/Cluster.h"

ClassImp(o2::TPC::Cluster)

using namespace o2::TPC;

//________________________________________________________________________
Cluster::Cluster()
  : Cluster(-1, -1, -1, -1, -1, -1, -1, -1)
{
}

//________________________________________________________________________
Cluster::Cluster(short cru, short row, float q, float qmax,
		 float padmean, float padsigma,
		 float timemean, float timesigma)
  : mCRU(cru)
  , mRow(row)
  , mQ(q)
  , mQmax(qmax)
  , mPadMean(padmean)
  , mPadSigma(padsigma)
  , mTimeMean(timemean)
  , mTimeSigma(timesigma)
  , FairTimeStamp()
{
}

//________________________________________________________________________
Cluster::Cluster(const Cluster& other)
  : mCRU(other.mCRU)
  , mRow(other.mRow)
  , mQ(other.mQ)
  , mQmax(other.mQmax)
  , mPadMean(other.mPadMean)
  , mPadSigma(other.mPadSigma)
  , mTimeMean(other.mTimeMean)
  , mTimeSigma(other.mTimeSigma)
  , FairTimeStamp(other)
{
}

//________________________________________________________________________
void Cluster::setParameters(short cru, short row, float q, float qmax,
			    float padmean, float padsigma,
			    float timemean, float timesigma)
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
std::ostream& Cluster::Print(std::ostream& out) const {
//std::ostream &Cluster::Print(std::ostream &output) const
//{
  out << "TPC Cluster in CRU [" << mCRU << "], pad row ["
	 << mRow << "] with charge/maxCharge " << mQ << "/" << mQmax
	 << " and coordinates (" << mPadMean << ", " << mTimeMean << ")"
	 << " and width (" << mPadSigma << ", " << mTimeSigma << ")";
  return out;
}


