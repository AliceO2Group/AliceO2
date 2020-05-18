// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_BOOSTOPTIONSRETRIEVER_H_
#define O2_FRAMEWORK_BOOSTOPTIONSRETRIEVER_H_

#include "Framework/ParamRetriever.h"
#include "Framework/ConfigParamSpec.h"

#include <boost/property_tree/ptree.hpp>

#include <string>
#include <vector>

#if __has_include(<Configuration/ConfigurationInterface.h>)
#include <Configuration/ConfigurationInterface.h>
#else
namespace o2::configuration
{
class ConfigurationInterface;
}
#endif

namespace o2::framework
{

/// ParamRetriever which uses AliceO2Group/Configuration to get the options
class ConfigurationOptionsRetriever : public ParamRetriever
{
 public:
  ConfigurationOptionsRetriever(std::vector<ConfigParamSpec> const& schema,
                                configuration::ConfigurationInterface* cfg,
                                std::string const& mainKey);

  bool isSet(const char* name) const final;
  int getInt(const char* name) const final;
  int64_t getInt64(const char* name) const final;
  float getFloat(const char* name) const final;
  double getDouble(const char* name) const final;
  bool getBool(const char* name) const final;
  std::string getString(const char* name) const final;
  boost::property_tree::ptree getPTree(const char* name) const final;

 private:
  configuration::ConfigurationInterface const* mCfg;
  boost::property_tree::ptree mStore;
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_BOOSTOPTIONSRETRIEVER_H_
