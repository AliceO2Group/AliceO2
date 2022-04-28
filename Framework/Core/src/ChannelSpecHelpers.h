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
#ifndef O2_FRAMEWORK_CHANNELSPECHELPERS_H_
#define O2_FRAMEWORK_CHANNELSPECHELPERS_H_

#include "Framework/ChannelSpec.h"
#include <iosfwd>
#include <string_view>
#include <vector>

namespace o2::framework
{

/// Handler to parse the description of the --channel-config
struct FairMQChannelConfigParser {
  virtual void beginChannel() {}
  virtual void endChannel() {}
  virtual void property(std::string_view /* key */, std::string_view /* value */) {}
  virtual void error() {}
};

/// A parser which creates an OutputChannelSpec from the --channel-config
struct OutputChannelSpecConfigParser : FairMQChannelConfigParser {
  void beginChannel() override;
  void endChannel() override;
  void property(std::string_view /* key */, std::string_view /* value */) override;
  void error() override;
  std::vector<OutputChannelSpec> specs;
  int channelCount = 0;
};

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
  /// Parse @a channelConfig option, invoking the correct method of
  /// @a parser
  static void parseChannelConfig(char const* channelConfig, FairMQChannelConfigParser& parser);
  static std::string defaultIPCFolder();
};

/// Stream operators so that we can use ChannelType with Boost.Test
std::ostream& operator<<(std::ostream& s, ChannelType const& type);
/// Stream operators so that we can use ChannelString with Boost.Test
std::ostream& operator<<(std::ostream& s, ChannelMethod const& method);

} // namespace o2::framework

#endif // O2_FRAMEWORK_CHANNELSPECHELPERS_H_
