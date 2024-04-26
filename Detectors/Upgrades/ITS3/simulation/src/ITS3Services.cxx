// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ITS3Services.h
/// \brief Definition of the ITS3Services class
/// \author Fabrizio Grosa <fgrosa@cern.ch>

#include "ITS3Simulation/ITS3Services.h"

#include <fairlogger/Logger.h> // for LOG

namespace o2::its3
{

void ITS3Services::createCYSSAssembly(TGeoVolume* motherVolume)
{
  // Return the whole assembly
  LOGP(info, "Creating CYSS Assembly and attaching to {}", motherVolume->GetName());
}

} // namespace o2::its3
