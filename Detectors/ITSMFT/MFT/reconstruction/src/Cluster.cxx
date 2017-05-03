/// \file Cluster.cxx
/// \brief Implementation of the Cluster class
/// \author bogdan.vulpescu@cern.ch 
/// \date 03/05/2017

#include "MFTReconstruction/Cluster.h"

using namespace o2::MFT;

ClassImp(o2::MFT::Cluster)

//_____________________________________________________________________________
Cluster::Cluster() 
 : o2::ITSMFT::Cluster()
{

  // default constructor

}

//_____________________________________________________________________________
Cluster::~Cluster()
= default;

//_____________________________________________________________________________
Cluster::Cluster(const Cluster& cluster)
  : o2::ITSMFT::Cluster(cluster)
{

  // copy constructor

}
