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

#ifndef ALICEO2_ITS3_ITS3SERVICES_H
#define ALICEO2_ITS3_ITS3SERVICES_H

#include "TGeoVolume.h"

namespace o2::its3
{
/// This class defines the Geometry for the ITS3 services using TGeo.
class ITS3Services
{
 public:
  void createCYSSAssembly(TGeoVolume* motherVolume);

 private:
  ClassDefNV(ITS3Services, 0); // ITS3 services
};

} // namespace o2::its3

#endif
