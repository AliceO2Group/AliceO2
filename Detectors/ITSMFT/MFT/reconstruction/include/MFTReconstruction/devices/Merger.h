#ifndef MERGER_H_
#define MERGER_H_

#include "TClonesArray.h"
#include "TFile.h"
#include "TFolder.h"
#include "TTree.h"
#include "FairEventHeader.h"

#include <FairMQDevice.h>

namespace AliceO2 {

namespace MFT {

class EventHeader;

typedef std::multimap<std::pair<std::pair<int,int>,int>,TObject*> MultiMapDef;

class Merger : public FairMQDevice
{

 public:

  Merger();
  virtual ~Merger();
  
  void SetNofParts(int iparts) { mNofParts = iparts; }
  
 protected:

  virtual void Init();
  bool MergeData(FairMQParts&, int);
  
 private:
  
  EventHeader* mEventHeader;
  int mNofParts;
  
  std::map<std::pair<int,int>,int> mNofPartsPerEventMap;  // number of parts for pair<event number,run id>
  MultiMapDef mObjectMap;            // TObjects for given pair<pair<event number, run,id>part>
  
  std::pair<int, int> mEvRIPair;
  std::pair<std::pair<int,int>,int> mEvRIPartTrio;
  std::pair<MultiMapDef::iterator, MultiMapDef::iterator> mRet;

  std::string mInputChannelName;
  std::string mOutputChannelName;

  int mNofReceivedMessages;
  int mNofSentMessages;

  Merger(const Merger&);
  Merger& operator=(const Merger&);
  
};
 
}
}

#endif
