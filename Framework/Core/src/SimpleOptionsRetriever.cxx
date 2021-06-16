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

#include "Framework/SimpleOptionsRetriever.h"
#include "Framework/ConfigParamSpec.h"

#include "PropertyTreeHelpers.h"

#include <boost/property_tree/ptree.hpp>

#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>

using namespace o2::framework;
namespace bpo = boost::program_options;

namespace o2::framework
{

void SimpleOptionsRetriever::update(std::vector<ConfigParamSpec> const& schema,
                                    boost::property_tree::ptree& store,
                                    boost::property_tree::ptree& provenance)
{
  PropertyTreeHelpers::populate(schema, store, mTree, provenance, mProvenanceLabel);
}

} // namespace o2::framework
