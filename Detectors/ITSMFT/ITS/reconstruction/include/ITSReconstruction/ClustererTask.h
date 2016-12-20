/// \file ClustererTask.h
/// \brief Definition of the ITS cluster finder task

#ifndef ALICEO2_ITS_CLUSTERERTASK
#define ALICEO2_ITS_CLUSTERERTASK

#include "FairTask.h" // for FairTask, InitStatus
#include "Rtypes.h"   // for ClustererTask::Class, ClassDef, etc

#include "ITSReconstruction/Clusterer.h"

class TClonesArray;

namespace AliceO2
{
namespace ITS
{
class ClustererTask : public FairTask
{
 public:
  ClustererTask();
  virtual ~ClustererTask();

  virtual InitStatus Init();
  virtual void Exec(Option_t* option);

 private:
  Clusterer mClusterer; ///< Cluster finder

  TClonesArray* mDigitsArray;   ///< Array of digits
  TClonesArray* mClustersArray; ///< Array of clusters

  ClassDef(ClustererTask, 2)
};
}
}

#endif /* ALICEO2_ITS_CLUSTERERTASK */
