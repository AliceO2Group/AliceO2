// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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

  TClonesArray*     mClusters;         
  TClonesArray*     mTracks;  

  Int_t mNClusters;
  Int_t mNTracks;     

  Int_t mTNofEvents;
  Int_t mTNofClusters;
  Int_t mTNofTracks;

  EventHeader *mEventHeader;

  ClassDefOverride(FindTracks,1);

};

}
}

#endif
