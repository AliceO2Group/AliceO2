// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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
