/// \file Cluster.h
/// \brief Cluster structure for TPC
#ifndef ALICEO2_TPC_CLUSTER_H
#define ALICEO2_TPC_CLUSTER_H

#ifndef __CINT__
#include <boost/serialization/base_object.hpp>  // for base_object
#endif

#include "FairTimeStamp.h"                      // for FairTimeStamp
#include "Rtypes.h"                             // for Double_t, ULong_t, etc
#include "TMath.h"
namespace boost { namespace serialization { class access; } }

namespace AliceO2{
  namespace TPC{
    
    /// \class Cluster
    /// \brief Cluster class for the TPC
    ///
    class Cluster : public FairTimeStamp {
    public:
      
      /// Default constructor
      Cluster();
      
      /// Constructor, initializing all values
      /// @param cru CRU (sector)
      /// @param row Row
      /// @param q Total charge of cluster
      /// @param qmax Maximum charge in a single cell (pad, time)
      /// @param padmean Mean position of cluster in pad direction
      /// @param padsigma Sigma of cluster in pad direction
      /// @param timemean Mean position of cluster in time direction
      /// @param timesigma Sigma of cluster in time direction
      Cluster(Short_t cru, Short_t row, Float_t q, Float_t qmax, 
	      Float_t padmean, Float_t padsigma, Float_t timemean, 
	      Float_t timesigma);
      
      /// Destructor
      virtual ~Cluster();

      /// Copy Constructor
      Cluster(const Cluster& other);


      friend bool operator==(const Cluster& lhs, const Cluster& rhs) {
        if (lhs.getRow()        != rhs.getRow())        return false;
        if (lhs.getCRU()        != rhs.getCRU())        return false;
        if (lhs.getQmax()       != rhs.getQmax())       return false;
        if (lhs.getQ()          != rhs.getQ())          return false;
        if (lhs.getPadMean()    != rhs.getPadMean())    return false;
        if (lhs.getTimeMean()   != rhs.getTimeMean())   return false;
        return true;
      }

      bool sim(const Cluster& rhs) {
        if (mRow        != rhs.getRow())        return false;
        if (mCRU        != rhs.getCRU())        return false;
        if (mQmax       != rhs.getQmax())       return false;
        if (mQ          != rhs.getQ())          return false;
        if (TMath::Abs(mPadMean - rhs.getPadMean()) > diffThreshold)    return false;
        if (TMath::Abs(mTimeMean - rhs.getTimeMean()) > diffThreshold)   return false;
        return true;
      }


      void setParameters(Short_t cru, Short_t row, Float_t q, Float_t qmax, 
			 Float_t padmean, Float_t padsigma,
			 Float_t timemean, Float_t timesigma);
            
      Int_t getCRU() const { return mCRU; }
      Int_t getRow() const { return mRow; }
      Float_t getQ() const { return mQ; }
      Float_t getQmax() const { return mQmax; }
      Float_t getPadMean() const { return mPadMean; }
      Float_t getTimeMean() const { return mTimeMean; }
      Float_t getPadSigma() const { return mPadSigma; }
      Float_t getTimeSigma() const { return mTimeSigma; }
      
      /// Print function: Print basic digit information on the  output stream
      /// @param output Stream to put the digit on
      /// @return The output stream
      friend std::ostream& operator<< (std::ostream& out, const Cluster &c) { return c.Print(out); }; 
//      std::ostream& Print(std::ostream &output) const;

    protected:      
      std::ostream& Print(std::ostream& out) const;
      
    private:
#ifndef __CINT__
      friend class boost::serialization::access;
#endif

      static constexpr Double_t diffThreshold = 0.0001;
      
      Short_t   mCRU;
      Short_t   mRow;
      Float_t   mQ;
      Float_t   mQmax;
      Float_t   mPadMean;
      Float_t   mPadSigma;
      Float_t   mTimeMean;
      Float_t   mTimeSigma;
            
      ClassDef(Cluster, 1);
    };
  }
}

#endif /* ALICEO2_TPC_CLUSTER_H */   
