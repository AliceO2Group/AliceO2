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

class DataDescriptorMatcher;

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
  static std::shared_ptr<DataDescriptorMatcher> buildFromKeepConfig(std::string const& config);
};

} // namespace framework
} // namespace o2
#endif // o2_framework_DataDescriptorQueryBuilder_H_INCLUDED
