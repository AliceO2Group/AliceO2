// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_CHANNELSPECHELPERS_H
#define FRAMEWORK_CHANNELSPECHELPERS_H

#include "Framework/ChannelSpec.h"
#include <iosfwd>

namespace o2
{
namespace framework
{

/// A few helpers to convert enums to their actual representation in
/// configuration files / GUI / string based APIs. Never too late
/// for C++ to get multimethods.
struct ChannelSpecHelpers {
  /// return a ChannelType as a lowercase string
  static char const* typeAsString(enum ChannelType type);
  /// return a ChannelMethod as a lowercase string
  static char const* methodAsString(enum ChannelMethod method);
  /// @return a url associated to an InputChannelSpec
  static std::string channelUrl(InputChannelSpec const&);
  /// @return a url associated to an OutputChannelSpec
  static std::string channelUrl(OutputChannelSpec const&);
};

/// Stream operators so that we can use ChannelType with Boost.Test
std::ostream& operator<<(std::ostream& s, ChannelType const& type);
/// Stream operators so that we can use ChannelString with Boost.Test
std::ostream& operator<<(std::ostream& s, ChannelMethod const& method);

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_CHANNELSPECHELPERS_H
