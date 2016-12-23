/// \file CookedTrackerTask.h
/// \brief Definition of the ITS "Cooked Matrix" tracker task
/// \author iouri.belikov@cern.ch

#ifndef ALICEO2_ITS_COOKEDTRACKERTASK_H
#define ALICEO2_ITS_COOKEDTRACKERTASK_H

#include "Rtypes.h"   
#include "FairTask.h" 

#include "ITSBase/GeometryTGeo.h"
#include "ITSReconstruction/CookedTracker.h"

class TClonesArray;

namespace AliceO2
{
namespace ITS
{
class CookedTrackerTask : public FairTask
{
 public:
  CookedTrackerTask();
  virtual ~CookedTrackerTask();

  virtual InitStatus Init();
  virtual void Exec(Option_t* option);

 private:
  GeometryTGeo mGeometry; ///< ITS geometry
  CookedTracker mTracker; ///< Track finder

  const TClonesArray* mClustersArray;   ///< Array of clusters
  TClonesArray* mTracksArray; ///< Array of tracks

  ClassDef(CookedTrackerTask, 1)
};
}
}

#endif /* ALICEO2_ITS_COOKEDTRACKERTASK */
