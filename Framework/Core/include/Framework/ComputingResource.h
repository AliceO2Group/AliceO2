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
#ifndef O2_FRAMEWORK_COMPUTINGRESOURCE_H_
#define O2_FRAMEWORK_COMPUTINGRESOURCE_H_

#include <string>

namespace o2::framework
{

struct ComputingOffer {
  float cpu = 0;
  float memory = 0;
  std::string hostname = "";
  unsigned short startPort = 0;
  unsigned short rangeSize = 0;
};

/// A computing resource which can be offered to run a device
struct ComputingResource {
  ComputingResource() = default;
  ComputingResource(ComputingOffer const& offer)
    : cpu(offer.cpu),
      memory(offer.memory),
      hostname(offer.hostname),
      startPort(offer.startPort),
      lastPort(offer.startPort)
  {
  }

  float cpu = 0;
  float memory = 0;
  std::string hostname = "";
  unsigned short startPort = 0;
  unsigned short lastPort = 0;
  unsigned short usedPorts = 0;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_COMPUTINGRESOURCES_H_
