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
#ifndef O2_FRAMEWORK_CONFIGPARAMSTORE_H_
#define O2_FRAMEWORK_CONFIGPARAMSTORE_H_

#include "Framework/ParamRetriever.h"
#include "Framework/ConfigParamSpec.h"

#include <boost/property_tree/ptree_fwd.hpp>

namespace o2::framework
{

/// This provides unified store for the parameters specified in the
/// ConfigParamSpecs. This provides only the store, not the actual
/// API to access it. Notice how the loading of the data is done in
/// to steps, to allow doing a diff between the old and the new
/// configuration.
class ConfigParamStore
{
 public:
  ConfigParamStore(std::vector<ConfigParamSpec> const& specs,
                   std::vector<std::unique_ptr<ParamRetriever>> retrievers);

  /// Preload the next store with a new copy of
  /// the configuration.
  void preload();

  void load(std::vector<ConfigParamSpec>& specs);

  /// Get the store
  boost::property_tree::ptree& store() { return *mStore; };
  boost::property_tree::ptree& provenanceTree() { return *mProvenance; };

  /// Get the specs
  [[nodiscard]] std::vector<ConfigParamSpec> const& specs() const
  {
    return mSpecs;
  }

  /// Activate the next store
  void activate();

  std::string provenance(const char*) const;

 private:
  std::vector<ConfigParamSpec> const& mSpecs;
  std::vector<std::unique_ptr<ParamRetriever>> mRetrievers;
  std::unique_ptr<boost::property_tree::ptree> mStore;
  std::unique_ptr<boost::property_tree::ptree> mProvenance;
  std::unique_ptr<boost::property_tree::ptree> mNextStore;
  std::unique_ptr<boost::property_tree::ptree> mNextProvenance;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_CONFIGPARAMREGISTRY_H_
