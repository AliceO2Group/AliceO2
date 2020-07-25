// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <iostream>
#include "DataFormatsTRD/LinkRecord.h"

namespace o2
{

namespace trd
{

void LinkRecord::printStream(std::ostream& stream) const
{
  stream << "Data for link 0x" << std::hex << mLinkId << std::dec << ", starting from entry " << getFirstEntry() << " with " << getNumberOfObjects() << " objects";
}

std::ostream& operator<<(std::ostream& stream, const LinkRecord& trg)
{
  trg.printStream(stream);
  return stream;
}

} // namespace trd
} // namespace o2
