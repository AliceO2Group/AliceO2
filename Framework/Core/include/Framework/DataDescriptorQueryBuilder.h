// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef o2_framework_DataDescriptorQueryBuilder_H_INCLUDED
#define o2_framework_DataDescriptorQueryBuilder_H_INCLUDED

#include <string>
#include <vector>
#include <memory>

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
  /// string := [a-zA-Z0-9_]*
  /// origin := string
  /// description := string
  /// subspec := [0-9]*
  /// spec := origin/description/subspec
  /// config := spec;spec;...
  ///
  /// Example for config: TPC/CLUSTER/0;ITS/TRACKS/1
  static DataDescriptorQuery buildFromKeepConfig(std::string const& config);
};

} // namespace framework
} // namespace o2
#endif // o2_framework_DataDescriptorQueryBuilder_H_INCLUDED
