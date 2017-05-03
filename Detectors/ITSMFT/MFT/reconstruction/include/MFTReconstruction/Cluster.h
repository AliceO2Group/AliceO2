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
