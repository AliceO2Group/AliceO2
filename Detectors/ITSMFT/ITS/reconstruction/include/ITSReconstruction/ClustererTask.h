//
//  ClustererTask.h
//

#ifndef _ALICEO2_ITS_ClustererTask_
#define _ALICEO2_ITS_ClustererTask_

#include "FairTask.h"  // for FairTask, InitStatus
#include "Rtypes.h"    // for ClustererTask::Class, ClassDef, etc

#include "ITSReconstruction/Clusterer.h"

class TClonesArray;

namespace AliceO2 {
  namespace ITS {
    
    class ClustererTask : public FairTask {
    public:
      ClustererTask();
      virtual ~ClustererTask();
      
      virtual InitStatus Init();
      virtual void Exec(Option_t *option);
      
    private:
      Clusterer            fClusterer;
      
      TClonesArray        *fDigitsArray;
      TClonesArray        *fClustersArray;
      
      ClassDef(ClustererTask, 2)
    };
  }
}

#endif
