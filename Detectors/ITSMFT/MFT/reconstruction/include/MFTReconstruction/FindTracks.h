/// \file FindTracks.h
/// \brief Simple track finding from the hits
/// \author bogdan.vulpescu@cern.ch 
/// \date 07/10/2016

#ifndef ALICEO2_MFT_FINDTRACKS_H_
#define ALICEO2_MFT_FINDTRACKS_H_

#include "FairTask.h"

class TClonesArray;

namespace o2 {
namespace MFT {

class EventHeader;

class FindTracks : public FairTask
{

 public:

  FindTracks();
  FindTracks(Int_t iVerbose);
  ~FindTracks() override;

  InitStatus Init() override;
  InitStatus ReInit() override;
  void Exec(Option_t* opt) override;

  void reset();

  virtual void initMQ(TList* tempList);
  virtual void execMQ(TList* inputList,TList* outputList);

 private:

  TClonesArray*     mHits;         
  TClonesArray*     mTracks;  

  Int_t mNHits;
  Int_t mNTracks;     

  Int_t mTNofEvents;
  Int_t mTNofHits;
  Int_t mTNofTracks;

  EventHeader *mEventHeader;

  ClassDefOverride(FindTracks,1);

};

}
}

#endif
