// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef o2_framework_PropertyTreeHelpers_H_INCLUDED
#define o2_framework_PropertyTreeHelpers_H_INCLUDED

#include "Framework/ConfigParamSpec.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/program_options/variables_map.hpp>

namespace o2
{
namespace framework
{

/// Helpers to manipulate property_trees
struct PropertyTreeHelpers {
  /// For all the options specified in @a schama, this fills
  /// @a tree with the contents of @a vmap, which is populated via boost
  /// program options.
  static void populate(std::vector<ConfigParamSpec> const& schema, boost::property_tree::ptree& tree, boost::program_options::variables_map const& vmap);
};

} // namespace framework
} // namespace o2

#endif // o2_framework_PropertyTreeHelpers_H_INCLUDED
