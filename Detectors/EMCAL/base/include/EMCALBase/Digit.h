#ifndef ALICEO2_EMCAL_DIGIT_H_
#define ALICEO2_EMCAL_DIGIT_H_

#include "FairTimeStamp.h"
#include "Rtypes.h"
#include <iosfwd>

#ifndef __CINT__
#include <boost/serialization/base_object.hpp>  // for base_object
#endif


namespace o2 {
  namespace EMCAL {
    
    /// \class Digit
    /// \brief EMCAL digit implementation
    class Digit : public FairTimeStamp {
    public:
      Digit() = default;
      
      Digit(Int_t module, Int_t tower, Double_t amplitude, Double_t time);
      ~Digit() override = default;
      
      bool operator<(const Digit &ref) const;
      
      Int_t GetModule() const { return mModule; }
      
      Int_t GetTower() const { return mTower; }
      
      Double_t GetAmplitude() const { return mAmplitude; }
      
      void SetModule(Int_t module) { mModule = module; }
      
      void SetTower(Int_t tower) { mTower = tower; }
      
      void SetAmplitude(Double_t amplitude) { mAmplitude = amplitude; }
      
      void PrintStream(std::ostream &stream) const;
      
    private:
#ifndef __CINT__
      friend class boost::serialization::access;
#endif
      
      Int_t             mModule;                ///< Supermodule index
      Int_t             mTower;                 ///< Tower index inside supermodule
      Double_t          mAmplitude;             ///< Amplitude
      
      ClassDefOverride(Digit, 1);
    };
    
    std::ostream &operator<<(std::ostream &stream, const Digit &dig);
  }
}
#endif
