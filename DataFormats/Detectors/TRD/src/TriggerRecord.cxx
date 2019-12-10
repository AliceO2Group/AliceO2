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
#include "DataFormatsTRD/TriggerRecord.h"

namespace o2
{

namespace trd
{

void TriggerRecord::printStream(std::ostream& stream) const
{
  stream << "Data for bc " << getBCData().bc << ", orbit " << getBCData().orbit << ", starting from entry " << getFirstEntry() << " with " << getNumberOfObjects() << " objects";
}

std::ostream& operator<<(std::ostream& stream, const TriggerRecord& trg)
{
  trg.printStream(stream);
  return stream;
}

} // namespace trd
} // namespace o2
