// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_OUTPUTREF_H
#define FRAMEWORK_OUTPUTREF_H

#include "Headers/DataHeader.h"

#include <string>

namespace o2
{
namespace framework
{

/// A reference to an output spec
///
/// OutputRef descriptors are expected to be passed as rvalue, i.e. a temporary object in the
/// function call. This is due to the fact that the header stack has no ordinary copy
/// constructor but only a move constructor
struct OutputRef {
  std::string label;
  header::DataHeader::SubSpecificationType subSpec;
  header::Stack headerStack = {};

  OutputRef(std::string&& l, header::DataHeader::SubSpecificationType s = 0) : label(std::move(l)), subSpec(s) {}

  OutputRef(const std::string& l, header::DataHeader::SubSpecificationType s = 0) : label(l), subSpec(s) {}

  OutputRef(std::string&& l, header::DataHeader::SubSpecificationType s, o2::header::Stack&& stack)
    : label(std::move(l)), subSpec(s), headerStack(std::move(stack))
  {
  }
};

} // namespace framework
} // namespace o2
#endif
