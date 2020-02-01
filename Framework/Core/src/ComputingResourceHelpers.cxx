// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "ComputingResourceHelpers.h"
#include <thread>
#include <unistd.h>
#include <sstream>

namespace o2::framework
{
long getTotalNumberOfBytes()
{
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  return pages * page_size;
};

ComputingResource ComputingResourceHelpers::getLocalhostResource()
{
  ComputingResource result;
  result.cpu = std::thread::hardware_concurrency(),
  result.memory = getTotalNumberOfBytes();
  result.hostname = "localhost";
  result.startPort = 22000;
  result.lastPort = 23000;
  result.usedPorts = 0;
  return result;
}

std::vector<ComputingResource> ComputingResourceHelpers::parseResources(std::string const& resourceString)
{
  std::vector<ComputingResource> resources;
  std::istringstream str{resourceString};
  std::string result;
  while (std::getline(str, result, ',')) {
    std::istringstream in{result};
    char colon;
    ComputingResource resource;
    std::getline(in, resource.hostname, ':');
    in >> resource.cpu >> colon >> resource.memory >> colon >> resource.startPort >> colon >> resource.lastPort;
    resource.memory = resource.memory * 1000000;
    resource.usedPorts = 0;
    resources.emplace_back(resource);
  }
  return resources;
}

} // namespace o2::framework
