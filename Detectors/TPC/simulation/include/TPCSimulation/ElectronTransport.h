/// \file ElectronTransport.h
/// \brief This class handles the transport of electrons in the active volume of the TPC
/// \author Andi Mathis, andreas.mathis@ph.tum.de
/// @todo Include distortions of the drift field

#ifndef ALICEO2_TPC_ElectronTransport_H_
#define ALICEO2_TPC_ElectronTransport_H_

#include "Rtypes.h"
#include "TPCSimulation/Constants.h"
#include "TPCBase/RandomRing.h"

namespace AliceO2 {
namespace TPC {
    
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
    void getElectronDrift(Float_t *posEle);
    
    /// Attachment probability for a given drift time
    /// @param driftTime Drift time of the electron
    /// @return Probability for attachment during the drift
    Float_t getAttachmentProbability(Float_t driftTime);
    
        
  private:
    RandomRing     mRandomGaus;        ///<  Circular random buffer containing random values of the Gauss distribution to take into account diffusion of the electrons
};

// Inline implementation

inline
Float_t ElectronTransport::getAttachmentProbability(Float_t driftTime)
{
  return ATTCOEF * OXYCONT * driftTime;
}

}
}

#endif // ALICEO2_TPC_ElectronTransport_H_
