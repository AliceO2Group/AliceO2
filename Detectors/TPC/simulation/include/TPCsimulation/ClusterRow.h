#ifndef _ALICEO2_TPC_ClusterRow_
#define _ALICEO2_TPC_ClusterRow_

#include "Rtypes.h"

class TClonesArray;

namespace AliceO2 {
    namespace TPC {
        
        class Cluster;
	
        class ClusterRow{
        public:
	  ClusterRow(Int_t mRowID);
	  ~ClusterRow();
            
            void Reset();
            
            Int_t GetRow() {return mRowID;}
            
            void SetCluster(Int_t pad, Int_t time, Float_t charge);
            
            void FillOutputContainer(TClonesArray *output, Int_t cruID, Int_t rowID);
            
        private:
            Short_t               mRow;           ///< Row
            std::vector<Cluster*> mClusters;
        };
    }
}

#endif
