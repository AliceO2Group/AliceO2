// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/FairOptionsRetriever.h"
#include "Framework/ConfigParamSpec.h"
#include "PropertyTreeHelpers.h"

#include <options/FairMQProgOptions.h>
#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <string>
#include <vector>

using namespace o2::framework;
namespace bpo = boost::program_options;
namespace bpt = boost::property_tree;

namespace o2::framework
{

void FairOptionsRetriever::update(std::vector<ConfigParamSpec> const& schema,
                                  boost::property_tree::ptree& store,
                                  boost::property_tree::ptree& provenance)
{
  PropertyTreeHelpers::populate(schema, store, mOpts->GetVarMap(), provenance);
}

} // namespace o2::framework
