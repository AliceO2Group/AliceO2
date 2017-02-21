/// \file ElectronTransport.h
/// \brief This class handles the transport of electrons in the active volume of the TPC
/// \author Andi Mathis, andreas.mathis@ph.tum.de
/// @todo Include distortions of the drift field

#ifndef ALICEO2_TPC_ElectronTransport_H_
#define ALICEO2_TPC_ElectronTransport_H_

#include "TPCSimulation/Constants.h"

#include "TPCBase/RandomRing.h"
#include "TPCBase/Mapper.h"

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
    /// @param posEle GlobalPosition3D with start position of the electrons
    /// @return GlobalPosition3D with position of the electrons after the drift taking into account diffusion
    GlobalPosition3D getElectronDrift(GlobalPosition3D posEle);

    /// Attachment probability for a given drift time
    /// @param driftTime Drift time of the electron
    /// @return Boolean whether the electron is attached (and lost) or not
    bool isElectronAttachment(float driftTime);


  private:
    RandomRing     mRandomGaus;   ///< Circular random buffer containing random values of the Gauss distribution to take into account diffusion of the electrons
    RandomRing     mRandomFlat;   ///< Circular random buffer containing flat random values to take into account electron attachement during drift
};

inline
bool ElectronTransport::isElectronAttachment(float driftTime)
{
  if(mRandomFlat.getNextValue() < ATTCOEF * OXYCONT * driftTime) {
    return true;        //electron is attached and lost
  }
  else return false;    // not attached
}
}
}

#endif // ALICEO2_TPC_ElectronTransport_H_
