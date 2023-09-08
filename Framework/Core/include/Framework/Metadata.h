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
#ifndef O2_FRAMEWORK_METADATA_H_
#define O2_FRAMEWORK_METADATA_H_
#include <string>

namespace o2::framework
{
/// Metadata for the analysis framework.
/// You need to declare it as a member of your AnalysisTask
/// like you do for configurable parameters. The constructor
/// allows you to subscribe to a given key, and the value
/// will be read from the data file and provided to you via
/// the get() method or by using the dereference operator.
/// Notice that for the moment metadata is provided as a string,
/// we might provide polymorphic access if people ask for it.
/// \example
/// \code{.cpp}
/// struct MyTask : AnalysisTask {
///  Metadata<std::string> someMetadata{"label"};
///  ...
///  void process(aod::Tracks const& tracks) {
///    std::cout << someMetadata.get() << std::endl;
///  }
/// };
/// \endcode
struct Metadata {
  std::string key;
  std::string value;

  Metadata(std::string key)
    : key(std::move(key))
  {
  }

  [[nodiscard]] std::string_view get() const
  {
    return this->value;
  }

  operator std::string_view() const
  {
    return this->value;
  }

  std::string_view const operator*() const
  {
    return this->value;
  }

  std::string_view const operator->() const
  {
    return this->value;
  }
};

/// Can be used to group together a number of Configurables
/// to overcome the limit of 100 Configurables per task.
/// In order to do so you can do:
///
/// struct MyTask {
///   struct : MetadataGroup {
///     Metadata<std::string> someMetadata{"label"};
///   } group;
/// };
///
/// and access it with
///
/// group.someMetadata;
struct MetadataGroup {
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_METADATA_H_
