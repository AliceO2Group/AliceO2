// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_RAWDEVICESERVICE_H
#define FRAMEWORK_RAWDEVICESERVICE_H

#include <map>
#include <string>
#include <vector>

namespace o2 {
namespace framework {

/// This service provides a hook into the actual fairmq device running the
/// computation, and allows and advanced user to modify its behavior in
/// from with a workflow class. This should be used to implement special
/// `DataProcessors` like one that acts as a gateway to standard FairMQ
/// devices.
class RawDeviceService {
public:
  virtual FairMQDevice *device() = 0;
  virtual void setDevice(FairMQDevice *device) = 0;
};

} // framework
} // o2
#endif // FRAMEWORK_RAWDEVICESERVICE_H
