/// \file GEMAmplification.h
/// \brief This class handles the amplification of electrons in the 4-GEM stack
/// \author Andi Mathis, andreas.mathis@ph.tum.de
#ifndef _ALICEO2_TPC_GEMAmplification_H_
#define _ALICEO2_TPC_GEMAmplification_H_

#include "Rtypes.h"
#include "TPCSimulation/Constants.h"
#include "TPCBase/RandomRing.h"

namespace AliceO2 {
namespace TPC {
    
/// \class GEMAmplification
/// \brief Class taking care of the amplification of electrons in the GEM stack
    
class GEMAmplification
{
  public:
      
    /// Default constructor
    GEMAmplification();
      
    /// Constructor 
    /// @param effGainGEM1 Effective gain of GEM 1
    /// @param effGainGEM2 Effective gain of GEM 2
    /// @param effGainGEM3 Effective gain of GEM 3
    /// @param effGainGEM4 Effective gain of GEM 4
    GEMAmplification(Float_t effGainGEM1, Float_t effGainGEM2, Float_t effGainGEM3, Float_t effGainGEM4);

    /// Destructor
    ~GEMAmplification();
   
    /// Compute the number of electrons after amplification in a full stack of four GEM foils
    /// @return Number of electrons after amplification in a full stack of four GEM foils
    Int_t getStackAmplification();
      
    /// Compute the number of electrons after amplification in a single GEM foil
    /// @param nElectrons Number of electrons to be amplified
    /// @param GEMgain Effective gain of that specific GEM
    /// @return Number of electrons after amplification in a single GEM foil
    Int_t getSingleGEMAmplification(Int_t nElectrons, Float_t GEMgain);
      
  private:
    Float_t        mEffGainGEM1;      ///<  Effective gain of GEM 1
    Float_t        mEffGainGEM2;      ///<  Effective gain of GEM 2
    Float_t        mEffGainGEM3;      ///<  Effective gain of GEM 3
    Float_t        mEffGainGEM4;      ///<  Effective gain of GEM 4
    RandomRing     mRandomPolya;      ///<  Circular random buffer containing random values of the polya distribution for gain fluctuations in a single GEM
      
    ClassDef(GEMAmplification, 1);
};
    
inline
Int_t GEMAmplification::getSingleGEMAmplification(Int_t nElectrons, Float_t GEMgain)
{
  // the incoming number of electrons from the foil above is multiplied 
  // by the effective gain and the fluctuations which follow a Polya
  // distribution
  return static_cast<int>(static_cast<float>(nElectrons)*GEMgain*mRandomPolya.getNextValue());
}

  
}
}

#endif
