// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_CONFIGPARAMREGISTRY_H
#define FRAMEWORK_CONFIGPARAMREGISTRY_H

#include "Framework/ParamRetriever.h"

#include <boost/property_tree/ptree.hpp>
#include <memory>
#include <string>
#include <cassert>

namespace o2
{
namespace framework
{

/// This provides unified access to the parameters specified in the workflow
/// specification.
/// The ParamRetriever is a concrete implementation of the registry which
/// will actually get the options. For example it could get them from the
/// FairMQ ProgOptions plugin or (to run "device-less", e.g. in batch simulation
/// jobs).
/// FIXME: Param is a bad name as FairRoot uses it for conditions data.
///        Use options? YES! OptionsRegistry...
class ConfigParamRegistry
{
public:
  ConfigParamRegistry(std::unique_ptr<ParamRetriever> retriever)
  : mRetriever{std::move(retriever)} {
  }

  template <typename T>
  typename std::enable_if_t<std::is_constructible<T, boost::property_tree::ptree>::value == false, T>
    get(const char*) const
  {
    static_assert(std::is_constructible<T, boost::property_tree::ptree>::value == false,
                  "No constructor from ptree provided");
  }

  /// Generic getter to extract an object of type T.
  /// Notice that in order for this to work you need to have
  /// a constructor which takes a ptree.
  template <typename T>
  std::enable_if_t<std::is_constructible<T, boost::property_tree::ptree>::value, T>
    get(const char* key) const
  {
    return T{ mRetriever->getPTree(key) };
  }

 private:
  std::unique_ptr<ParamRetriever> mRetriever;
};

template <> inline int ConfigParamRegistry::get<int>(const char *key) const {
  assert(mRetriever.get());
  return mRetriever->getInt(key);
}

template <> inline float ConfigParamRegistry::get<float>(const char *key) const {
  assert(mRetriever.get());
  return mRetriever->getFloat(key);
}

template <> inline double ConfigParamRegistry::get<double>(const char *key) const {
  assert(mRetriever.get());
  return mRetriever->getDouble(key);
}

template <> inline std::string ConfigParamRegistry::get<std::string>(const char *key) const {
  assert(mRetriever.get());
  return mRetriever->getString(key);
}

template <> inline bool ConfigParamRegistry::get<bool>(const char *key) const {
  assert(mRetriever.get());
  return mRetriever->getBool(key);
}

template <>
inline boost::property_tree::ptree ConfigParamRegistry::get<boost::property_tree::ptree>(const char* key) const
{
  assert(mRetriever.get());
  return mRetriever->getPTree(key);
}

} // namespace framework
} // namespace o2

#endif //FRAMEWORK_CONFIGPARAMREGISTRY_H
