/// \file HwCluster.h
/// \brief Class to have some more info about the HwClusterer clusters
#ifndef ALICEO2_TPC_HWCLUSTER_H
#define ALICEO2_TPC_HWCLUSTER_H

//#ifndef __CINT__
//#include <boost/serialization/base_object.hpp>  // for base_object
//#endif

#include "TPCSimulation/Cluster.h"
#include "Rtypes.h"                             // for Double_t, ULong_t, etc
#include "TPCSimulation/HwFixedPoint.h" 

namespace boost { namespace serialization { class access; } }

namespace AliceO2{
  namespace TPC{

    class HwCluster : public Cluster {
    public:

      // Constructors
      HwCluster(
          Short_t sizeP = 5, 
          Short_t sizeT = 5,
          UShort_t fpTotalWidth = 23,
          UShort_t fpDecPrec = 12);
      HwCluster(
          Short_t cru, Short_t row, 
          Short_t sizeP, Short_t sizeT, 
          Float_t** clusterData, 
          Short_t maxPad, Short_t maxTime,
          UShort_t fpTotalWidth = 23,
          UShort_t fpDecPrec = 12);

      // Destructor
      ~HwCluster();

      // Copy Constructor
      HwCluster(const HwCluster& other);

      Short_t getPad()      const { return mPad; }
      Short_t getTime()     const { return mTime; }
      Short_t getSizeP()    const { return mSizeP; }
      Short_t getSizeT()    const { return mSizeT; }

      void setClusterData(Short_t cru, Short_t row, Short_t sizeP, Short_t sizeT, 
                          Float_t** clusterData, Short_t maxPad, Short_t maxTime);
      void calculateClusterProperties(Short_t cru, Short_t row);
      friend std::ostream& operator<< (std::ostream& out, const HwCluster &c) { return c.Print(out); }
      std::ostream& Print(std::ostream &output) const;
      std::ostream& PrintDetails(std::ostream &output) const;
//      void Print();

    private:

      Bool_t        mPropertiesCalculatd;
      Short_t       mPad;
      Short_t       mTime;
      Short_t       mSizeP;
      Short_t       mSizeT;
      Short_t       mSize;

      UShort_t mFixedPointTotalWidth;
      UShort_t mFixedPointDecPrec;

      HwFixedPoint**     mClusterData;

      ClassDef(HwCluster, 1);
    };
  }
}

#endif
