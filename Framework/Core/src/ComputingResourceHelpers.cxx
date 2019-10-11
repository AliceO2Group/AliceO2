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

namespace o2::framework
{
long getTotalNumberOfBytes()
{
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  return pages * page_size;
};

ComputingResource ComputingResourceHelpers::getLocalhostResource(unsigned short startPort, unsigned short rangeSize)
{
  ComputingResource result;
  result.cpu = std::thread::hardware_concurrency(),
  result.memory = getTotalNumberOfBytes();
  result.hostname = "localhost";
  result.startPort = startPort;
  result.lastPort = startPort + rangeSize;
  result.usedPorts = 0;
  return result;
}
} // namespace o2::framework
