/// \file FindHits.h
/// \brief Simple hits finding from the points
/// \author bogdan.vulpescu@cern.ch 
/// \date 10/10/2016

#ifndef ALICEO2_MFT_FINDHITS_H_
#define ALICEO2_MFT_FINDHITS_H_

#include "FairTask.h"

class FairMCEventHeader;

class TClonesArray;

namespace AliceO2 {
namespace MFT {

class EventHeader;

class FindHits : public FairTask
{

 public:

  FindHits();
  virtual ~FindHits();

  void Reset();

  virtual InitStatus Init();
  virtual InitStatus ReInit();
  virtual void Exec(Option_t* opt);

  virtual void InitMQ(TList* tempList);
  virtual void ExecMQ(TList* inputList,TList* outputList);

 private:

  FindHits(const FindHits&);
  FindHits& operator=(const FindHits&);

  TClonesArray* mPoints; //!
  TClonesArray* mHits;   //!

  Int_t mNHits;

  Int_t mTNofEvents;
  Int_t mTNofHits;

  FairMCEventHeader *mMCEventHeader;
  EventHeader *mEventHeader;

  ClassDef(FindHits,1);

};

}
}

#endif
