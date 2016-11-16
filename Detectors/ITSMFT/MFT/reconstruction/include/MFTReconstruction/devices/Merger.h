#ifndef MERGER_H_
#define MERGER_H_

#include "TClonesArray.h"
#include "TFile.h"
#include "TFolder.h"
#include "TTree.h"
#include "FairEventHeader.h"

#include "FairMQDevice.h"

#include "MFTBase/EventHeader.h"

namespace AliceO2 {

namespace MFT {

class Merger : public FairMQDevice
{

 public:

  Merger();
  virtual ~Merger();
  
  void SetNofParts(int iparts) { fNofParts = iparts; }
  
 protected:

  virtual void Init();
  virtual void Run();
  
 private:
  
  EventHeader* fEventHeader;
  int fNofParts;
  
  std::map     <std::pair<int,int>,int>                     fNofPartsPerEventMap;  // number of parts for pair<event number,run id>
  std::multimap<std::pair<std::pair<int,int>,int>,TObject*> fObjectMap;            // TObjects for given pair<pair<event number, run,id>part>
  
  Merger(const Merger&);
  Merger& operator=(const Merger&);
  
};
 
}
}

#endif
