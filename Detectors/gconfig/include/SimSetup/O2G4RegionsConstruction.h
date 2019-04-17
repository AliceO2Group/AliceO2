// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file G4RegionsConstruction.h
/// \brief Definition of the Regions to be created by GEANT4 class

#ifndef DETECTORS_GCONFIG_INCLUDE_SIMSETUP_O2G4REGIONSCONSTRUCTION_H_
#define DETECTORS_GCONFIG_INCLUDE_SIMSETUP_O2G4REGIONSCONSTRUCTION_H_

#include "TG4VUserRegionConstruction.h"

namespace o2
{

class O2G4RegionsConstruction : public TG4VUserRegionConstruction
{
  public:
    O2G4RegionsConstruction() = default;
    ~O2G4RegionsConstruction() override = default;

    // Override this final
    void Construct() final;

  private:
    /// Not implemented
    O2G4RegionsConstruction(const O2G4RegionsConstruction& right);
    /// Not implemented
    O2G4RegionsConstruction& operator=(const O2G4RegionsConstruction& right);


};

} // namespace o2

#endif /* DETECTORS_GCONFIG_INCLUDE_SIMSETUP_G4REGIONSCONSTRUCTION_H_ */
