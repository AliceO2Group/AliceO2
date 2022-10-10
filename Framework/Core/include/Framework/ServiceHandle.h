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
#ifndef O2_FRAMEWORK_SERVICEHANDLE_H_
#define O2_FRAMEWORK_SERVICEHANDLE_H_

#include <string>

namespace o2::framework
{

/// The kind of service we are asking for
enum struct ServiceKind {
  /// A Service which is not thread safe, therefore all accesses to it must be mutexed.
  Serial,
  /// A Service which is thread safe and therefore can be used by many threads the same time without risk
  Global,
  /// A Service which is specific to a given thread in a thread pool
  Stream
};

struct ServiceTypeHash {
  uint32_t hash = 0;
};

/// Handle to the service hash must be calculated
/// using TypeIdHelper::uniqueId<BaseClass>() so that
/// we can retrieve the service by its baseclass.
struct ServiceHandle {
  /// Unique hash associated to the type of service.
  unsigned int hash;
  /// Type erased pointer to a service
  void* instance = nullptr;
  /// Kind of service
  ServiceKind kind = ServiceKind::Serial;
  /// Mnemonic name to use for the service.
  std::string name = "unknown";
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_SERVICEHANDLE_H_
