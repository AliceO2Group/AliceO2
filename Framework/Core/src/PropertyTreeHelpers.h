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
#ifndef O2_FRAMEWORK_PROPERTYTREEHELPERS_H_
#define O2_FRAMEWORK_PROPERTYTREEHELPERS_H_

#include "Framework/ConfigParamSpec.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/program_options/variables_map.hpp>
#include <functional>

namespace o2::framework
{

/// Helpers to manipulate property_trees.
struct PropertyTreeHelpers {
  /// For all the options specified in @a schama, this fills
  /// @a tree with the contents of @a vmap, which is populated via boost
  /// program options. Any key in the @a schema will be marked as
  /// "default" in the provenance ptree.
  static void populateDefaults(std::vector<ConfigParamSpec> const& schema,
                               boost::property_tree::ptree& tree,
                               boost::property_tree::ptree& provenance);

  /// For all the options specified in @a schama, this fills
  /// @a tree with the contents of @a vmap, which is populated via boost
  /// program options. Any key in the @a schema will be marked as
  /// "fairmq" in the provenance ptree.
  static void populate(std::vector<ConfigParamSpec> const& schema,
                       boost::property_tree::ptree& tree,
                       boost::program_options::variables_map const& vmap,
                       boost::property_tree::ptree& provenance);

  /// For all the options specified in @a schama, this fills
  /// @a tree with the contents of @a in, which is another ptree
  /// e.g. populated using ConfigurationInterface.
  /// Any key in the @a schema will be marked as "configuration" in the provenance ptree.
  static void populate(std::vector<ConfigParamSpec> const& schema,
                       boost::property_tree::ptree& tree,
                       boost::property_tree::ptree const& in,
                       boost::property_tree::ptree& provenance,
                       std::string const& propertyLabel);

  //using WalkerFunction = std::function<void(boost::property_tree::ptree const&, boost::property_tree::ptree::path_type, boost::property_tree::ptree const&)>;
  using WalkerFunction = std::function<void(boost::property_tree::ptree const&, boost::property_tree::ptree::path_type, boost::property_tree::ptree const&)>;
  /// Traverse the tree recursively calling @a WalkerFunction on each leaf.
  static void traverse(boost::property_tree::ptree const& parent, WalkerFunction& method);

  /// Merge @a source ptree into @a dest
  static void merge(boost::property_tree::ptree& dest,
                    boost::property_tree::ptree const& source,
                    boost::property_tree::ptree::path_type const& mergePoint);
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_PROPERTYTREEHELPERS_H_
