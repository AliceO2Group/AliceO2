// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_PROPERTYTREEHELPERS_H_
#define O2_FRAMEWORK_PROPERTYTREEHELPERS_H_

#include "Framework/ConfigParamSpec.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/program_options/variables_map.hpp>

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
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_PROPERTYTREEHELPERS_H_
