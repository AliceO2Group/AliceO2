// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef o2_framework_readers_AODReaderHelpers_INCLUDED_H
#define o2_framework_readers_AODReaderHelpers_INCLUDED_H

#include "Framework/TableBuilder.h"
#include "Framework/AlgorithmSpec.h"

namespace o2
{
namespace framework
{
namespace readers
{

struct AODReaderHelpers {
  static AlgorithmSpec rootFileReaderCallback();
  static AlgorithmSpec run2ESDConverterCallback();
};

} // namespace readers
} // namespace framework
} // namespace o2

#endif
