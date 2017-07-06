// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Cluster.h
/// \brief Simple hit obtained from points
/// \author bogdan.vulpescu@cern.ch 
/// \date 10/10/2016

#ifndef ALICEO2_MFT_CLUSTER_H_
#define ALICEO2_MFT_CLUSTER_H_

#include "ITSMFTReconstruction/Cluster.h"
#include "MFTBase/GeometryTGeo.h"

namespace o2 {
namespace MFT {

class Cluster : public o2::ITSMFT::Cluster
{

 public:
  
  Cluster();
  Cluster(const Cluster& cluster);
  Cluster& operator=(const Cluster& cluster) = delete;
  ~Cluster() override;
  
 private:
  
  ClassDefOverride(Cluster,1);

};

}
}

#endif
