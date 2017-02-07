/// \file GEMAmplification.h
/// \brief This class handles the amplification of electrons in the 4-GEM stack
/// \author Andi Mathis, andreas.mathis@ph.tum.de
#ifndef ALICEO2_TPC_GEMAmplification_H_
#define ALICEO2_TPC_GEMAmplification_H_

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
    GEMAmplification(float effGainGEM1, float effGainGEM2, float effGainGEM3, float effGainGEM4);

    /// Destructor
    ~GEMAmplification();
   
    /// Compute the number of electrons after amplification in a full stack of four GEM foils
    /// @return Number of electrons after amplification in a full stack of four GEM foils
    int getStackAmplification();
    
    /// Compute the number of electrons after amplification in a full stack of four GEM foils
    /// @param nElectrons Number of electrons arriving at the first amplification stage (GEM1)
    /// @return Number of electrons after amplification in a full stack of four GEM foils
    int getStackAmplification(int nElectrons);
      
    /// Compute the number of electrons after amplification in a single GEM foil
    /// @param nElectrons Number of electrons to be amplified
    /// @param GEMgain Effective gain of that specific GEM
    /// @return Number of electrons after amplification in a single GEM foil
    int getSingleGEMAmplification(int nElectrons, float GEMgain);
      
  private:
    float          mEffGainGEM1;      ///<  Effective gain of GEM 1
    float          mEffGainGEM2;      ///<  Effective gain of GEM 2
    float          mEffGainGEM3;      ///<  Effective gain of GEM 3
    float          mEffGainGEM4;      ///<  Effective gain of GEM 4
    RandomRing     mRandomPolya;      ///<  Circular random buffer containing random values of the polya distribution for gain fluctuations in a single GEM
};
    
inline
int GEMAmplification::getSingleGEMAmplification(int nElectrons, float GEMgain)
{
  // the incoming number of electrons from the foil above is multiplied 
  // by the effective gain and the fluctuations which follow a Polya
  // distribution
  return static_cast<int>(static_cast<float>(nElectrons)*GEMgain*mRandomPolya.getNextValue());
}

  
}
}

#endif // ALICEO2_TPC_GEMAmplification_H_
