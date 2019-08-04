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
    : mRetriever{std::move(retriever)}
  {
  }

  template <typename T>
  T get(const char* key) const
  {
    assert(mRetriever);
    try {
      if constexpr (std::is_same_v<T, int>) {
        return mRetriever->getInt(key);
      } else if constexpr (std::is_same_v<T, float>) {
        return mRetriever->getFloat(key);
      } else if constexpr (std::is_same_v<T, double>) {
        return mRetriever->getDouble(key);
      } else if constexpr (std::is_same_v<T, std::string>) {
        return mRetriever->getString(key);
      } else if constexpr (std::is_same_v<T, bool>) {
        return mRetriever->getBool(key);
      } else if constexpr (std::is_same_v<T, boost::property_tree::ptree>) {
        return mRetriever->getPTree(key);
      } else if constexpr (std::is_constructible_v<T, boost::property_tree::ptree>) {
        return T{mRetriever->getPTree(key)};
      } else if constexpr (std::is_constructible_v<T, boost::property_tree::ptree> == false) {
        static_assert(std::is_constructible_v<T, boost::property_tree::ptree> == false,
                      "Not a basic type and no constructor from ptree provided");
      }
    } catch (std::exception& e) {
      throw std::invalid_argument(std::string("missing option: ") + key + " (" + e.what() + ")");
    } catch (...) {
      throw std::invalid_argument(std::string("error parsing option: ") + key);
    }
  }

 private:
  std::unique_ptr<ParamRetriever> mRetriever;
};

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_CONFIGPARAMREGISTRY_H
