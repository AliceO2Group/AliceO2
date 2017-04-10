/// \file BoxClusterer.h
/// \brief Class for TPC cluster finding
#ifndef ALICEO2_TPC_BoxClusterer_H_
#define ALICEO2_TPC_BoxClusterer_H_

#include "Rtypes.h"
#include "TPCSimulation/Clusterer.h"

class TClonesArray;

namespace o2{
  
  namespace TPC {
    
    class ClusterContainer;
    
    class BoxClusterer : public Clusterer {
    public:
      BoxClusterer();
      
      /// Destructor
      ~BoxClusterer();
      
      // Should this really be a public member?
      // Maybe better to just call by process
      void Init() override;
      
      /// Steer conversion of points to digits
      /// @param digits Container with TPC digits
      /// @return Container with clusters
      ClusterContainer* Process(TClonesArray *digits) override;
      
    private:
      // To be done
      /* BoxClusterer(const BoxClusterer &); */
      /* BoxClusterer &operator=(const BoxClusterer &); */
      
      void FindLocalMaxima(const Int_t iCRU);
      void CleanArrays();
      void GetPadAndTimeBin(Int_t bin, Short_t& iPad, Short_t& iTimeBin);
      Int_t Update(const Int_t iCRU, const Int_t iRow, const Int_t iPad, 
		   const Int_t iTimeBin, Float_t signal);
      Float_t GetQ(const Float_t* adcArray, const Short_t pad,
           const Short_t time, Short_t& timeMin, Short_t& timeMax, 
		   Short_t& padMin, Short_t& padMax) const;
      Bool_t UpdateCluster(Float_t charge, Int_t deltaPad, Int_t deltaTime, 
			   Float_t& qTotal, Double_t& meanPad, 
			   Double_t& sigmaPad, Double_t& meanTime, 
			   Double_t& sigmaTime);
      
      
      //
      //  Expand buffer
      //
      Float_t** mAllBins;      //!<! Array for digit using random access
      Int_t**   mAllSigBins;   //!<! Array of pointers to the indexes over threshold
      Int_t*    mAllNSigBins;  //!<! Array with number of signals in each row
      
      ClassDefNV(BoxClusterer, 1);
    };
  }
}


#endif 
