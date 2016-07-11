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
class TClonesArray;
namespace AliceO2 { namespace TPC { class Clusterer; } }  // lines 19-19

namespace AliceO2 {
  namespace TPC{
    
    class BoxClusterer;
    
    class ClustererTask : public FairTask{
    public:
      ClustererTask();
      virtual ~ClustererTask();
      
      virtual InitStatus Init();
      virtual void Exec(Option_t *option);
      
      //             Clusterer *GetClusterer() const { return fClusterer; }
      
    private:
      BoxClusterer        *fClusterer;
      
      TClonesArray        *fDigitsArray;
      TClonesArray        *fClustersArray;
      
      ClassDef(ClustererTask, 1)
        };
  }
}

#endif
