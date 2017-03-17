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
    /// taking into account collection and extraction efficiencies and fluctuations of the GEM amplification
    /// @param nElectrons Number of electrons to be amplified
    /// @param GEM Number of the GEM in the stack (1, 2, 3, 4)
    /// @return Number of electrons after amplification in a single GEM foil
    int getSingleGEMAmplification(int nElectrons, int GEM);
      
    /// Compute the electron losses due to extraction or collection efficiencies
    /// @param nElectrons Input number of electrons
    /// @param probability Collection or extraction efficiency
    /// @return Number of electrons after probable losses
    int getElectronLosses(int nElectrons, float probability);

    /// Compute the number of electrons after amplification in a single GEM foil
    /// taking into account avalanche fluctuations (Polya for <500 electrons and Gaus (central limit theorem) for a larger number of electrons)
    /// @param nElectrons Input number of electrons
    /// @param GEM Number of the GEM in the stack (1, 2, 3, 4)
    /// @return Number of electrons after amplification in the GEM
    int getGEMMultiplication(int nElectrons, int GEM);

  private:
    RandomRing     mRandomGaus;       ///< Circular random buffer containing random Gaus values for gain fluctuation if the number of electrons is larger (central limit theorem)
    RandomRing     mRandomFlat;       ///< Circular random buffer containing flat random values for the collection/extraction
    std::array<RandomRing, 4> mGain;  ///< Container with random Polya distributions, one for each GEM in the stack
};
  
}
}

#endif // ALICEO2_TPC_GEMAmplification_H_
