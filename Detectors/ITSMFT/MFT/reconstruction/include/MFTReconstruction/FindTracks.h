/// \file FindTracks.h
/// \brief Simple track finding from the hits
/// \author bogdan.vulpescu@cern.ch 
/// \date 07/10/2016

#ifndef ALICEO2_MFT_FINDTRACKS_H_
#define ALICEO2_MFT_FINDTRACKS_H_

#include "FairTask.h"

class TClonesArray;

namespace AliceO2 {
namespace MFT {

class EventHeader;

class FindTracks : public FairTask
{

 public:

  FindTracks();
  FindTracks(Int_t iVerbose);
  virtual ~FindTracks();

  void Reset();

  virtual InitStatus Init();
  virtual InitStatus ReInit();
  virtual void Exec(Option_t* opt);

  virtual void InitMQ(TList* tempList);
  virtual void ExecMQ(TList* inputList,TList* outputList);

 private:

  TClonesArray*     fHits;         
  TClonesArray*     fTracks;  

  Int_t fNHits;
  Int_t fNTracks;     

  Int_t fTNofEvents;
  Int_t fTNofHits;
  Int_t fTNofTracks;

  EventHeader *fEventHeader;

  ClassDef(FindTracks,1);

};

}
}

#endif
