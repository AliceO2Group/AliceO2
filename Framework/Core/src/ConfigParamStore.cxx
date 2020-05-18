// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/ConfigParamStore.h"
#include "Framework/ParamRetriever.h"
#include "PropertyTreeHelpers.h"
#include <boost/property_tree/ptree.hpp>

namespace o2::framework
{

ConfigParamStore::ConfigParamStore(std::vector<ConfigParamSpec> const& specs,
                                   std::vector<std::unique_ptr<ParamRetriever>> retrievers)
  : mSpecs(specs),
    mRetrievers{std::move(retrievers)},
    mStore{new boost::property_tree::ptree{}},
    mProvenance{new boost::property_tree::ptree{}},
    mNextStore{new boost::property_tree::ptree{}},
    mNextProvenance{new boost::property_tree::ptree{}}
{
}

/// Preload the next store with a new copy of
/// the configuration.
void ConfigParamStore::preload()
{
  mNextStore->clear();
  mNextProvenance->clear();
  // By default we populate with code.
  PropertyTreeHelpers::populateDefaults(mSpecs, *mNextStore, *mNextProvenance);
  for (auto& retriever : mRetrievers) {
    retriever->update(mSpecs, *mNextStore, *mNextProvenance);
  }
}

/// Activate the next store
void ConfigParamStore::activate()
{
  mStore->swap(*mNextStore);
  mProvenance->swap(*mNextProvenance);
}

std::string ConfigParamStore::provenance(const char* key) const
{
  return mProvenance->get<std::string>(key);
}
} // namespace o2::framework
