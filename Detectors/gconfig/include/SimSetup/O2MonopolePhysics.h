// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author M+Giacalone - July 2026

#ifndef O2_G4CONFIG_O2MONOPOLEPHYSICS_H_
#define O2_G4CONFIG_O2MONOPOLEPHYSICS_H_

#include <TString.h>

class TG4RunConfiguration;

namespace o2
{
namespace g4config
{

/// Build a Geant4-VMC run configuration which attaches the magnetic-monopole ionisation process
/// (G4mplIonisation) to the requested reference physics list.
/// the monopole particles already defined by O2 (PDG +-4110000 / +-4120000).
///
/// Only used when G4Params.monopole == true;
/// By default, the standard TG4RunConfiguration is created
///
/// \param userGeometry        VMC geometry-navigation string (as for TG4RunConfiguration)
/// \param physicsList         the reference physics-list selection string
/// \param specialProcess      the VMC special-process selection string
/// \param specialStacking     VMC special-stacking flag
/// \param mtApplication       multithreading flag
/// \param magneticChargeDirac magnetic charge of the monopole in units of the
///                            Dirac charge g_D = eplus/(2*alpha) (~68.5 eplus);
///                            1.0 (default) reproduces the classic single Dirac monopole
TG4RunConfiguration* createMonopoleRunConfiguration(const TString& userGeometry,
                                                    const TString& physicsList,
                                                    const TString& specialProcess,
                                                    bool specialStacking,
                                                    bool mtApplication,
                                                    double magneticChargeDirac);

} // namespace g4config
} // namespace o2

#endif // O2_G4CONFIG_O2MONOPOLEPHYSICS_H_
