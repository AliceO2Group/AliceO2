/// \file ClustererTask.h
/// \brief Definition of the ITS cluster finder task

#ifndef ALICEO2_ITS_CLUSTERERTASK
#define ALICEO2_ITS_CLUSTERERTASK

#include "FairTask.h" 

#include "ITSBase/GeometryTGeo.h"
#include "ITSReconstruction/Clusterer.h"

class TClonesArray;

namespace o2
{
namespace ITS
{
class ClustererTask : public FairTask
{
 public:
  ClustererTask();
  ~ClustererTask() override;

  InitStatus Init() override;
  void Exec(Option_t* option) override;

 private:
  GeometryTGeo mGeometry; ///< ITS geometry
  Clusterer mClusterer;   ///< Cluster finder

  TClonesArray* mDigitsArray;   ///< Array of digits
  TClonesArray* mClustersArray; ///< Array of clusters

  ClassDefOverride(ClustererTask, 1)
};
}
}

#endif /* ALICEO2_ITS_CLUSTERERTASK */
