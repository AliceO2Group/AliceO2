//
//  ClustererTask.h
//  ALICEO2
//
//
//

#ifndef __ALICEO2__ClustererTask__
#define __ALICEO2__ClustererTask__

#include <cstdio>
#include "FairTask.h"  // for FairTask, InitStatus
#include "Rtypes.h"    // for ClustererTask::Class, ClassDef, etc
#include "TPCSimulation/Clusterer.h"       // for Clusterer
#include "TPCSimulation/BoxClusterer.h"       // for Clusterer
#include "TPCSimulation/HwClusterer.h"       // for Clusterer

class TClonesArray;

namespace o2 {
  namespace TPC{
    
    class ClustererTask : public FairTask{
    public:
      ClustererTask();
      ~ClustererTask() override;
      
      InitStatus Init() override;
      void Exec(Option_t *option) override;

      enum class ClustererType : int { HW, Box};
      void setClustererEnable(ClustererType type, bool val) {
        switch (type) {
          case ClustererType::HW:   mHwClustererEnable = val; break;
          case ClustererType::Box:  mBoxClustererEnable = val; break;
        };
      };

      bool isClustererEnable(ClustererType type) const { 
        switch (type) {
          case ClustererType::HW:   return mHwClustererEnable;
          case ClustererType::Box:  return mBoxClustererEnable;
        };
      };

      Clusterer* getClusterer(ClustererType type) { 
        switch (type) {
          case ClustererType::HW:   return mHwClusterer;
          case ClustererType::Box:  return mBoxClusterer;
        };
      };
      
      BoxClusterer* getBoxClusterer()   const { return mBoxClusterer; };
      HwClusterer* getHwClusterer()     const { return mHwClusterer; };
      //             Clusterer *GetClusterer() const { return fClusterer; }
      
    private:
      bool          mBoxClustererEnable;
      bool          mHwClustererEnable;

      BoxClusterer        *mBoxClusterer;
      HwClusterer         *mHwClusterer;
      
      TClonesArray        *mDigitsArray;
      TClonesArray        *mClustersArray;
      TClonesArray        *mHwClustersArray;
      
      ClassDefOverride(ClustererTask, 1)
    };
  }
}

#endif
