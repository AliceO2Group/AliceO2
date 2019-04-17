// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file O2G4RunConfiguration.h
/// \brief Overriding what is necessary for custom O2 version of TG4RunConfiguration

#ifndef DETECTORS_GCONFIG_INCLUDE_SIMSETUP_O2G4RUNCONFIGURATION_H_
#define DETECTORS_GCONFIG_INCLUDE_SIMSETUP_O2G4RUNCONFIGURATION_H_

#include <string>

#include "TG4RunConfiguration.h"

class G4VUserPhysicsList;
class TG4VUserRegionConstruction;
class TG4VUserFastSimulation;

namespace o2
{

class O2G4RunConfiguration : public TG4RunConfiguration
{
  public:
    O2G4RunConfiguration(const std::string& userGeometry = "geomRoot",
                         const std::string& physicsList = "QGSP_FTFP_BERT+optical",
                         const std::string& specialProcess = "stepLimiter+specialCuts",
                         bool specialStacking = true,
                         bool mtApplication = true);

    virtual ~O2G4RunConfiguration() override = default;

    virtual TG4VUserRegionConstruction* CreateUserRegionConstruction() override;
    virtual TG4VUserFastSimulation* CreateUserFastSimulation() override;

    void SetFastSimConstruction(TG4VUserFastSimulation* fastSimConstruction);

  private:
    /// Not implemented
    O2G4RunConfiguration();
    O2G4RunConfiguration(const O2G4RunConfiguration& right);
    O2G4RunConfiguration& operator=(const O2G4RunConfiguration& right);

  private:
    TG4VUserFastSimulation* mFastSimConstruction;

};

} // namespace o2

#endif /* DETECTORS_GCONFIG_INCLUDE_SIMSETUP_O2G4RUNCONFIGURATION_H_ */
