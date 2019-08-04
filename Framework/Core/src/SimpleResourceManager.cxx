// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "SimpleResourceManager.h"
#include "ComputingResource.h"
#include <exception>
#include <stdexcept>

namespace o2
{
namespace framework
{

/// The simplest implementation of this allocates mMaxPorts ports starting from
/// the mInitialPort. For now we still consider everything running on a single
/// machine.
std::vector<ComputingResource> SimpleResourceManager::getAvailableResources()
{
  std::vector<ComputingResource> result;
  if ((mInitialPort < 1025) || ((mInitialPort + mMaxPorts) > 65535)) {
    throw std::runtime_error("Invalid port number. Valid port range is 1025-65535");
  }
  // We insert them backwards for compatibility with the previous
  // way of assigning them.
  for (size_t i = mInitialPort + mMaxPorts - 1; i >= mInitialPort; --i) {
    result.push_back(ComputingResource{
      1.0,
      1.0,
      "localhost",
      static_cast<unsigned short>(i)});
  };
  return result;
}

} // namespace framework
} // namespace o2
