// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file O2G4FastSimConstruction.h
/// \brief Definition of the FastSims to be created by GEANT4

#ifndef DETECTORS_GCONFIG_INCLUDE_SIMSETUP_O2G4FASTSIMCONSTRUCTION_H_
#define DETECTORS_GCONFIG_INCLUDE_SIMSETUP_O2G4FASTSIMCONSTRUCTION_H_

#include <string>
#include <vector>

#include "TG4VUserFastSimulation.h"

namespace o2
{
namespace base
{
class DirectTransport;
}
}

namespace o2
{

class O2G4FastSimConstruction : public TG4VUserFastSimulation
{
  public:
    O2G4FastSimConstruction(const std::vector<std::string>& directTransportRegions = {});
    ~O2G4FastSimConstruction() override = default;

    void AddDirectTransportRegion(const std::string& name);

    // Override this final
    void Construct() final;

  private:
    /// Not implemented
    O2G4FastSimConstruction(const O2G4FastSimConstruction& right);
    /// Not implemented
    O2G4FastSimConstruction& operator=(const O2G4FastSimConstruction& right);

  private:
    // names of regions which a DirectTransport will be built for
    std::vector<std::string> mDirectTransportRegions;
    std::vector<base::DirectTransport*> mDirectTransportModels;
};

} // namespace o2

#endif /* DETECTORS_GCONFIG_INCLUDE_SIMSETUP_O2G4FASTSIMCONSTRUCTION_H_ */
