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
#ifndef FRAMEWORK_PARTREF_H
#define FRAMEWORK_PARTREF_H

#include <memory>

#include <fairmq/FwdDecls.h>

namespace o2
{
namespace framework
{

/// Reference to an inflight part.
struct PartRef {
  std::unique_ptr<fair::mq::Message> header;
  std::unique_ptr<fair::mq::Message> payload;
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_PARTREF_H
