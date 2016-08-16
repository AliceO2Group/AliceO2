//
//  ClusterCRU.h
//  ALICEO2
//
//  Created by Markus Fasel on 25.03.15.
//
//

#ifndef _ALICEO2_TPC_ClusterCRU_
#define _ALICEO2_TPC_ClusterCRU_

#include "Rtypes.h"

class TClonesArray;

namespace AliceO2 {
  namespace TPC {
    
    class Cluster;
    class ClusterRow;
    
    class ClusterCRU{
    public:
      ClusterCRU(Int_t mCRUID, Int_t nrows);
      ~ClusterCRU();
      
      void Reset();
      Int_t GetCRUID() {return mCRUID;}
      
      void SetCluster(Int_t row, Int_t pad, Int_t time, Float_t charge);
      
      void FillOutputContainer(TClonesArray *output, Int_t cruID);
            
    private:
      Int_t               mCRUID;           ///< CRU ID
      Int_t               mNRows;           ///< Number of rows in CRU
      std::vector <ClusterRow*>  mRows;    // check whether there is something in the container before writing into it
    };
  }
}

#endif
