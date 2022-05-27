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
#ifndef o2_framework_DataDescriptorQueryBuilder_H_INCLUDED
#define o2_framework_DataDescriptorQueryBuilder_H_INCLUDED

#include "Framework/InputSpec.h"

#include <string>
#include <vector>
#include <memory>
#include <regex>

namespace o2
{
namespace framework
{

namespace data_matcher
{
class DataDescriptorMatcher;
}

/// Struct describing a query.
/// @a variableNames is filled with the variables which are
/// referenced in the matcher string. We return it as part of the query so that it can
/// be eventually passed to a different query builder which wants to use the
/// same variable names.  Alternatively we could simply build the query in
/// one go and return the number of
/// variables required by the context. Not sure what's the best approach.
struct DataDescriptorQuery {
  std::vector<std::string> variableNames;
  std::shared_ptr<data_matcher::DataDescriptorMatcher> matcher;
};

/// Various utilities to manipulate InputSpecs
struct DataDescriptorQueryBuilder {
  /// Creates an inputspec from a configuration @a config string with the
  /// following grammar.
  ///
  /// string := [a-zA-Z0-9_*]*
  /// binding := string
  /// origin := string
  /// description := string
  /// subspec := [0-9]*
  /// spec := binding:origin/description/subspec
  /// config := spec;spec;...
  ///
  /// Example for config: x:TPC/CLUSTER/0;y:ITS/TRACKS/1
  ///
  /// FIXME: grammar has been extended, add documentation
  static std::vector<InputSpec> parse(const char* s = "");

  /// Internal method to build matcher list from a string of verified specs,
  /// the fixed and verified input format allows simple scanning, no state based
  /// parsing nor error handling implemented
  static DataDescriptorQuery buildFromKeepConfig(std::string const& config);
  /// deprecated?
  static DataDescriptorQuery buildFromExtendedKeepConfig(std::string const& config);
  static std::unique_ptr<data_matcher::DataDescriptorMatcher> buildNode(std::string const& nodeString);
  static std::smatch getTokens(std::string const& nodeString);
};

} // namespace framework
} // namespace o2
#endif // o2_framework_DataDescriptorQueryBuilder_H_INCLUDED
