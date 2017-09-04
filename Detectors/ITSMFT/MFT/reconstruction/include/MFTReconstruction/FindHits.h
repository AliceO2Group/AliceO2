// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file FindHits.h
/// \brief Simple hits finding from the points
/// \author bogdan.vulpescu@cern.ch 
/// \date 10/10/2016

#ifndef ALICEO2_MFT_FINDHITS_H_
#define ALICEO2_MFT_FINDHITS_H_

#include "FairTask.h"

class FairMCEventHeader;

class TClonesArray;

namespace o2 {
namespace MFT {

class EventHeader;

class FindHits : public FairTask
{

 public:

  FindHits();
  ~FindHits() override;

  InitStatus Init() override;
  InitStatus ReInit() override;
  void Exec(Option_t* opt) override;

  void reset();

  virtual void initMQ(TList* tempList);
  virtual void execMQ(TList* inputList,TList* outputList);

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

  ClassDefOverride(FindHits,1);

};

}
}

#endif
