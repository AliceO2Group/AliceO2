/// \file Cluster.cxx
/// \brief Cluster structure for TPC clusters

#include "TPCSimulation/Cluster.h"

ClassImp(o2::TPC::Cluster)

using namespace o2::TPC;

//________________________________________________________________________
std::ostream& Cluster::Print(std::ostream& out) const {
//std::ostream &Cluster::Print(std::ostream &output) const
//{
  out << "TPC Cluster in CRU [" << mCRU << "], pad row ["
	 << mRow << "] with charge/maxCharge " << mQ << "/" << mQmax
	 << " and coordinates (" << mPadMean << ", " << GetTimeStamp() << ")"
	 << " and width (" << mPadSigma << ", " << GetTimeStampError() << ")";
  return out;
}


