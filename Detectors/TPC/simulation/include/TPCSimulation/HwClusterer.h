/// \file HwClusterer.h
/// \brief Class for TPC HW cluster finding
#ifndef ALICEO2_TPC_HWClusterer_H_
#define ALICEO2_TPC_HWClusterer_H_

#include "Rtypes.h"
#include "TPCSimulation/Clusterer.h"
#include "FairTask.h"  // for FairTask, InitStatus

#include <vector>


class TClonesArray;

namespace AliceO2{
  
  namespace TPC {
    
    class ClusterContainer;
    class ClustererTask;
    class HwClusterFinder;
    class HwCluster;
    class Digit;
    
    class HwClusterer : public Clusterer {
    public:
      HwClusterer(Int_t globalTime = 0);
      
      /// Destructor
      ~HwClusterer();
      
      // Should this really be a public member?
      // Maybe better to just call by process
      void Init();
      
      /// Steer conversion of points to digits
      /// @param digits Container with TPC digits
      /// @return Container with clusters
      ClusterContainer* Process(TClonesArray *digits);

      enum class Processing : int { Sequential, Parallel};
      void setProcessingType(Processing processing)    { mProcessingType = processing; };   
      
    private:
      // To be done
      /* BoxClusterer(const BoxClusterer &); */
      /* BoxClusterer &operator=(const BoxClusterer &); */
      
      static void processDigits(
          const std::vector<std::vector<Digit*>>& digits, 
          const std::vector<std::vector<HwClusterFinder*>>& clusterFinder, 
                std::vector<HwCluster>& cluster, 
                Int_t iCRU,
                Int_t maxRows,
                Int_t maxPads, 
                Int_t maxTime,
                Bool_t enableCM);
      
//      HwClusterFinder**** mClusterFinder;
      //    CRU         Row         CF
      std::vector<std::vector<std::vector<HwClusterFinder*>>> mClusterFinder;
      std::vector<std::vector<std::vector<Digit*>>> mDigitContainer;

      std::vector<std::vector<HwCluster>> mClusterStorage;
      
      Processing    mProcessingType; 

      Int_t     mGlobalTime;
      Int_t     mCRUs;
      Float_t   mMinQDiff;
      Bool_t    mAssignChargeUnique;
      Int_t     mPadsPerCF;
      Int_t     mTimebinsPerCF;
      Int_t     mCfPerRow;
      Bool_t    mEnableCommonMode;
      
      ClassDef(HwClusterer, 1);
    };
  }
}


#endif 
