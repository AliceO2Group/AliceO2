// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_COMPLETIONPOLICYHELPERS_H
#define FRAMEWORK_COMPLETIONPOLICYHELPERS_H

#include "Framework/ChannelSpec.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/CompletionPolicy.h"

#include <functional>
#include <string>

namespace o2
{
namespace framework
{

/// Helper class which holds commonly used policies.
struct CompletionPolicyHelpers {
  /// Default Completion policy. When all the parts of a record have
  /// arrived, consume them.
  static CompletionPolicy consumeWhenAll();
  /// When any of the parts of the record have been received, consume them.
  static CompletionPolicy consumeWhenAny();
  /// When any of the parts of the record have been received, process them,
  /// without actually consuming them.
  static CompletionPolicy processWhenAny();
  /// Attach a given @a op to a device matching @name.
  static CompletionPolicy defineByName(std::string const& name, CompletionPolicy::CompletionOp op);
};

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_COMPLETIONPOLICYHELPERS_H
