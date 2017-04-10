/// \file CookedTrackerTask.h
/// \brief Definition of the ITS "Cooked Matrix" tracker task
/// \author iouri.belikov@cern.ch

#ifndef ALICEO2_ITS_COOKEDTRACKERTASK_H
#define ALICEO2_ITS_COOKEDTRACKERTASK_H

#include "FairTask.h" 

#include "ITSBase/GeometryTGeo.h"
#include "ITSReconstruction/CookedTracker.h"

class TClonesArray;

namespace o2
{
namespace ITS
{
class CookedTrackerTask : public FairTask
{
 public:
  CookedTrackerTask(Int_t nThreads=1);
  ~CookedTrackerTask() override;

  InitStatus Init() override;
  void Exec(Option_t* option) override;
  void setBz(Double_t bz) { mTracker.setBz(bz); }

 private:
  Int_t mNumOfThreads;    ///< Number of threads
  GeometryTGeo mGeometry; ///< ITS geometry
  CookedTracker mTracker; ///< Track finder

  const TClonesArray* mClustersArray;   ///< Array of clusters
  TClonesArray* mTracksArray; ///< Array of tracks

  ClassDefOverride(CookedTrackerTask, 1)
};
}
}

#endif /* ALICEO2_ITS_COOKEDTRACKERTASK */
