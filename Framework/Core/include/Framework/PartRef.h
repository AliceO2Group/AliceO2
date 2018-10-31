// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_PARTREF_H
#define FRAMEWORK_PARTREF_H

#include <memory>

class FairMQMessage;

namespace o2
{
namespace framework
{

/// Reference to an inflight part.
struct PartRef {
  std::unique_ptr<FairMQMessage> header;
  std::unique_ptr<FairMQMessage> payload;
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_PARTREF_H
