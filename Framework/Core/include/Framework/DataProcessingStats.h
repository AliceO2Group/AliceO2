// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef o2_framework_DataProcessingStats_H_INCLUDED
#define o2_framework_DataProcessingStats_H_INCLUDED

#include <cstdint>

namespace o2
{
namespace framework
{

struct DataProcessingStats {
  int pendingInputs;
  int incomplete;
  int inputParts;
};

} // namespace framework
} // namespace o2

#endif // o2_framework_DataProcessingStats_H_INCLUDED
