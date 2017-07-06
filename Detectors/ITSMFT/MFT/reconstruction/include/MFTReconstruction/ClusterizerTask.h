// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterizerTask.h
/// \brief Task driving the cluster finding from digits
/// \author bogdan.vulpescu@cern.ch 
/// \date 03/05/2017

#ifndef ALICEO2_MFT_CLUSTERIZERTASK_H_
#define ALICEO2_MFT_CLUSTERIZERTASK_H_

#include "FairTask.h"

#include "MFTReconstruction/Clusterizer.h"

class FairMCEventHeader;

class TClonesArray;

namespace o2 
{
  namespace MFT 
  {
    class EventHeader; 
    class ClusterizerTask : public FairTask
    {
      
    public:
      
      ClusterizerTask();
      ~ClusterizerTask() override;
      
      InitStatus Init() override;
      InitStatus ReInit() override;
      void Exec(Option_t* opt) override;
      
      void reset();
      
      virtual void initMQ(TList* tempList);
      virtual void execMQ(TList* inputList,TList* outputList);
      
    private:
      
      Clusterizer mClusterizer;   ///< Cluster finder
      
      TClonesArray* mDigits; //!
      TClonesArray* mClusters;   //!
      
      Int_t mNClusters;
      
      Int_t mTNofEvents;
      Int_t mTNofClusters;
      
      FairMCEventHeader *mMCEventHeader;
      EventHeader *mEventHeader;
      
      ClassDefOverride(ClusterizerTask,1);
      
    };    
  }
}

#endif
