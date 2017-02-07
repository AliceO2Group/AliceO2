/// \file ElectronTransport.h
/// \brief This class handles the transport of electrons in the active volume of the TPC
/// \author Andi Mathis, andreas.mathis@ph.tum.de
/// @todo Include distortions of the drift field

#ifndef ALICEO2_TPC_ElectronTransport_H_
#define ALICEO2_TPC_ElectronTransport_H_

#include "TPCSimulation/Constants.h"
#include "TPCBase/RandomRing.h"

namespace AliceO2 {
namespace TPC {
  
template <typename T>
T diffusion(T random, T width, T pos) {
  return (random*width) + pos;
}
    
/// \class ElectronTransport
/// \brief Class taking care of the transport of electrons in the active volume of the TPC
    
class ElectronTransport
{
  public:
      
    /// Default constructor
    ElectronTransport();

    /// Destructor
    ~ElectronTransport();
    
    /// Drift of electrons in electric field taking into account diffusion
    /// @param *xyz Array with 3d position of the electrons
    /// @return Array with 3d position of the electrons after the drift taking into account diffusion
    void getElectronDrift(float *posEle);
    
    /// Drift of electrons in electric field taking into account diffusion using Vc
    /// @param *xyz Array with 3d position of the electrons
    /// @return Array with 3d position of the electrons after the drift taking into account diffusion
    void getElectronDriftVc(float *posEle);
    
    /// Attachment probability for a given drift time
    /// @param driftTime Drift time of the electron
    /// @return Probability for attachment during the drift
    float getAttachmentProbability(float driftTime) const { return ATTCOEF * OXYCONT * driftTime; }
    
        
  private:
    RandomRing     mRandomGaus;        ///< Circular random buffer containing random values of the Gauss distribution to take into account diffusion of the electrons
    int            mVc_size;           ///< Size of the vectorization array
};
}
}

#endif // ALICEO2_TPC_ElectronTransport_H_
