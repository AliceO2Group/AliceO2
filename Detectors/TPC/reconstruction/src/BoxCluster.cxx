// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file BoxCluster.cxx
/// \brief Class to have some more info about the BoxClusterer clusters

#include "TPCReconstruction/BoxCluster.h"
#include "TPCReconstruction/Cluster.h"

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
void BoxCluster::setBoxParameters(Short_t pad, Short_t time, Short_t size)
{
  mPad = pad;
  mTime = time;
  mSize = size;
}

//________________________________________________________________________
std::ostream& BoxCluster::print(std::ostream &output) const
{
  Cluster::print(output);
  output << " centered at (pad, time) = " << mPad << ", " << mTime
	 << " covering " << Int_t(mSize/10)  << " pads and " << mSize%10
	 << " time bins";
  return output;
}
