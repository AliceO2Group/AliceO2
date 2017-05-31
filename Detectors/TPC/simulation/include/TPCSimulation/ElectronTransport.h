// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ElectronTransport.h
/// \brief Definition of the electron transport
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_ElectronTransport_H_
#define ALICEO2_TPC_ElectronTransport_H_

#include "TPCSimulation/Constants.h"

#include "TPCBase/RandomRing.h"
#include "TPCBase/Mapper.h"

namespace o2 {
namespace TPC {

/// \class ElectronTransport
/// This class handles the electron transport in the active volume of the TPC.
/// In particular, in deals with the diffusion of the charge cloud while drifting towards the readout chambers and the loss of electrons during that drift due to attachement.

class ElectronTransport
{
  public:

    /// Default constructor
    ElectronTransport();

    /// Destructor
    ~ElectronTransport();

    /// Drift of electrons in electric field taking into account diffusion
    /// \param posEle GlobalPosition3D with start position of the electrons
    /// \return GlobalPosition3D with position of the electrons after the drift taking into account diffusion
    GlobalPosition3D getElectronDrift(GlobalPosition3D posEle);

    /// Attachment probability for a given drift time
    /// \param driftTime Drift time of the electron
    /// \return Boolean whether the electron is attached (and lost) or not
    bool isElectronAttachment(float driftTime);


  private:
    /// Circular random buffer containing random values of the Gauss distribution to take into account diffusion of the electrons
    RandomRing     mRandomGaus;
    /// Circular random buffer containing flat random values to take into account electron attachement during drift
    RandomRing     mRandomFlat;
};

inline
bool ElectronTransport::isElectronAttachment(float driftTime)
{
  if(mRandomFlat.getNextValue() < ATTCOEF * OXYCONT * driftTime) {
    return true;        ///electron is attached and lost
  }
  else return false;    /// not attached
}
}
}

#endif // ALICEO2_TPC_ElectronTransport_H_
