//
//  ClustererTask.h
//

#ifndef _ALICEO2_ITS_ClustererTask_
#define _ALICEO2_ITS_ClustererTask_

#include "FairTask.h"  // for FairTask, InitStatus
#include "Rtypes.h"    // for ClustererTask::Class, ClassDef, etc

class TClonesArray;

namespace AliceO2 {
  namespace ITS {
    
    //class Clusterer;
    
    class ClustererTask : public FairTask {
    public:
      ClustererTask();
      virtual ~ClustererTask();
      
      virtual InitStatus Init();
      virtual void Exec(Option_t *option);
      
      //Clusterer *GetClusterer() const { return fClusterer; }
      
    private:
      //Clusterer        *fClusterer;
      
      TClonesArray        *fDigitsArray;
      TClonesArray        *fClustersArray;
      
      ClassDef(ClustererTask, 1)
    };
  }
}

#endif
