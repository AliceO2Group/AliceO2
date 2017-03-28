/// \file Clusterer.h
/// \brief Base class for TPC clusterer
#ifndef ALICEO2_TPC_Clusterer_H_
#define ALICEO2_TPC_Clusterer_H_

#include "Rtypes.h"
#include "TObject.h"

class TClonesArray;

namespace AliceO2{
  
  namespace TPC {
    
    class ClusterContainer;
    
    class Clusterer : public TObject {
    public:
      Clusterer(
          Int_t rowsMax = 18,
          Int_t padsMax = 138,
          Int_t timeBinsMax = 1024, //300,
          Int_t minQMax = 5,
          Bool_t requirePositiveCharge = kTRUE,
          Bool_t requireNeighbouringPad = kTRUE);
      
      /// Destructor
      ~Clusterer();
      
      // Should this really be a public member?
      // Maybe better to just call by process
      virtual void Init() = 0;
      
      /// @param digits Container with TPC digits
      /// @return Container with clusters
      virtual ClusterContainer* Process(TClonesArray *digits) = 0;

      void setRowsMax(Int_t val)                    { mRowsMax = val; };
      void setPadsMax(Int_t val)                    { mPadsMax = val; };
      void setTimeBinsMax(Int_t val)                { mTimeBinsMax = val; };
      void setMinQMax(Float_t val)                  { mMinQMax = val; };
      void setRequirePositiveCharge(Bool_t val)     { mRequirePositiveCharge = val; };
      void setRequireNeighbouringPad(Bool_t val)    { mRequireNeighbouringPad = val; };

      Int_t     getRowsMax()                    const { return mRowsMax; };
      Int_t     getPadsMax()                    const { return mPadsMax; };
      Int_t     getTimeBinsMax()                const { return mTimeBinsMax; };
      Float_t   getMinQMax()                    const { return mMinQMax; };
      Bool_t    hasRequirePositiveCharge()      const { return mRequirePositiveCharge; };
      Bool_t    hasRequireNeighbouringPad()     const { return mRequireNeighbouringPad; };
      
    protected:
      
      ClusterContainer* mClusterContainer; ///< Internal cluster storage
      
      Int_t     mRowsMax;           //!<! Maximum row number
      Int_t     mPadsMax;           //!<! Maximum pad number
      Int_t     mTimeBinsMax;       //!<! Maximum time bin
      Float_t   mMinQMax;           //|<| Minimun Qmax for cluster
      Bool_t    mRequirePositiveCharge;  //|<|If true, require charge > 0 (else all clusters are automatic 5x5)
      Bool_t    mRequireNeighbouringPad; //|<|If true, require 2+ pads minimum
      
      ClassDef(Clusterer, 1);
    };
  }
}


#endif 
