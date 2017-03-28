/// \file HwClusterFinder.h
/// \brief Class for TPC HW cluster finder
#ifndef ALICEO2_TPC_HWClusterFinder_H_
#define ALICEO2_TPC_HWClusterFinder_H_

#include "Rtypes.h"
#include "TObject.h"
#include <vector>

namespace AliceO2{
  
  namespace TPC {

    class HwCluster;
    
    class HwClusterFinder : public TObject {
    public:
      enum ClusterProcessingType { kCharge, kSlope };

      // Constructor
      HwClusterFinder(Short_t cru, Short_t row, Short_t id, 
          Short_t padOffset, Short_t pads=8, Short_t timebins=8,
          Float_t diffThreshold=0, Float_t chargeThreshold=5, Bool_t requirePositiveCharge=kTRUE);

      // Destructor
      ~HwClusterFinder();

      // Copy constructor
      HwClusterFinder(const HwClusterFinder& other);

      Bool_t AddTimebin(Float_t* timebin, UInt_t globalTime, Int_t length = 8);
      Bool_t AddTimebins(Int_t nBins, Float_t** timebins, UInt_t globalTimeOfLast, Int_t length = 8);
      void AddZeroTimebin(UInt_t globalTime = 0, Int_t lengt = 8);
      void PrintLocalStorage();
      void PrintLocalSlopes();

      void reset(UInt_t globalTimeAfterReset);

      
      // Getter functions
      Int_t                     getGlobalTimeOfLast() const             { return mGlobalTimeOfLast; }
      ClusterProcessingType     getProcessingType() const               { return mProcessingType; }
      Short_t                   getCRU() const                          { return mCRU; }
      Short_t                   getRow() const                          { return mRow; }
      Short_t                   getId() const                           { return mId; }
      Short_t                   getPadOffset() const                    { return mPadOffset; }
      Short_t                   getNpads() const                        { return mPads; }
      Short_t                   getNtimebins() const                    { return mTimebins; }
      Short_t                   getClusterSizeP() const                 { return mClusterSizePads; }
      Short_t                   getClusterSizeT() const                 { return mClusterSizeTime; }
      Float_t                   getDiffThreshold() const                { return mDiffThreshold; }
      Float_t                   getChargeThreshold() const              { return mChargeThreshold; }
      Bool_t                    getRequirePositiveCharge() const        { return mRequirePositiveCharge; }
      Bool_t                    getRequireNeighbouringPad() const       { return mRequireNeighbouringPad; }
      Bool_t                    getRequireNeighbouringTimebin() const   { return mRequireNeighbouringTimebin; }
      Bool_t                    getAutProcessing() const                { return mAutoProcessing; } 
      Bool_t                    getmAssignChargeUnique() const          { return mAssignChargeUnique; }
      HwClusterFinder*          getNextCF() const                       { return mNextCF; }
      std::vector<HwCluster>*   getClusterContainer()                   { return &clusterContainer; }

      // Setter functions
//      void  setGlobalTimeOfLast(Int_t val)  { mGlobalTimeOfLast = val; }
      void  setProcessingType(ClusterProcessingType val)    { mProcessingType = val; }
      void  setCRU(Short_t val)                             { mCRU = val; }
      void  setRow(Short_t val)                             { mRow = val; }
      void  setId(Short_t val)                              { mId = val; }
      void  setPadOffset(Short_t val)                       { mPadOffset = val; }
      void  setNpads(Short_t val)                           { mPads = val; }
      void  setNtimebins(Short_t val)                       { mTimebins = val; }
      void  setClusterSizeP(Short_t val)                    { mClusterSizePads = val; }
      void  setClusterSizeT(Short_t val)                    { mClusterSizeTime = val; }
      void  setDiffThreshold(Float_t val)                   { mDiffThreshold = val; }
      void  setChargeThreshold(Float_t val)                 { mChargeThreshold = val; }
      void  setRequirePositiveCharge(Bool_t val)            { mRequirePositiveCharge = val; }
      void  setRequireNeighbouringPad(Bool_t val)           { mRequireNeighbouringPad = val; }
      void  setRequireNeighbouringTimebin(Bool_t val)       { mRequireNeighbouringTimebin = val; }
      void  setAutoProcessing(Bool_t val)                   { mAutoProcessing = val; }
      void  setAssignChargeUnique(Bool_t val)               { mAssignChargeUnique = val; }
      void  setNextCF(HwClusterFinder* nextCF);


      void clearClusterContainer()        { clusterContainer.clear(); }

      Bool_t FindCluster();

      void clusterAlreadyUsed(Short_t time, Short_t pad, Float_t** cluster);

    private:

      Float_t chargeForCluster(Float_t* charge, Float_t* toCompare);
      void printCluster(Short_t time, Short_t pad);

      // local variables
      std::vector<HwCluster> clusterContainer;
      Int_t mTimebinsAfterLastProcessing;
      Float_t** mData;
      Float_t** mSlopesP;
      Float_t** mSlopesT;
      Float_t** tmpCluster;
      Float_t*  mZeroTimebin;


      // configuration
      ClusterProcessingType mProcessingType;
      Int_t mGlobalTimeOfLast;
      Short_t mCRU;
      Short_t mRow;
      Short_t mId;
      Short_t mPadOffset;
      Short_t mPads;
      Short_t mTimebins;
      Short_t mClusterSizePads;
      Short_t mClusterSizeTime;
      Float_t mDiffThreshold;
      Float_t mChargeThreshold;
      Bool_t mRequirePositiveCharge;
      Bool_t mRequireNeighbouringPad;
      Bool_t mRequireNeighbouringTimebin;
      Bool_t mAutoProcessing;
      Bool_t mAssignChargeUnique;

      HwClusterFinder* mNextCF;

      ClassDef(HwClusterFinder, 1);
    };
  }
}


#endif 
