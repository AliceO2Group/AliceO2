//
//  ClustererTask.h
//  ALICEO2
//
//
//

#ifndef __ALICEO2__ClustererTask__
#define __ALICEO2__ClustererTask__

#include <stdio.h>
#include "FairTask.h"  // for FairTask, InitStatus
#include "Rtypes.h"    // for ClustererTask::Class, ClassDef, etc
#include "TPCSimulation/Clusterer.h"       // for Clusterer
#include "TPCSimulation/BoxClusterer.h"       // for Clusterer
#include "TPCSimulation/HwClusterer.h"       // for Clusterer

class TClonesArray;

namespace AliceO2 {
  namespace TPC{
    
    class ClustererTask : public FairTask{
    public:
      ClustererTask();
      virtual ~ClustererTask();
      
      virtual InitStatus Init();
      virtual void Exec(Option_t *option);

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
          case ClustererType::HW:   return fHwClusterer;
          case ClustererType::Box:  return fBoxClusterer;
        };
      };
      
      BoxClusterer* getBoxClusterer()   const { return fBoxClusterer; };
      HwClusterer* getHwClusterer()     const { return fHwClusterer; };
      //             Clusterer *GetClusterer() const { return fClusterer; }
      
    private:
      bool          mBoxClustererEnable;
      bool          mHwClustererEnable;

      BoxClusterer        *fBoxClusterer;
      HwClusterer         *fHwClusterer;
      
      TClonesArray        *fDigitsArray;
      TClonesArray        *fClustersArray;
      TClonesArray        *fHwClustersArray;
      
      ClassDef(ClustererTask, 1)
    };
  }
}

#endif
