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
#include <memory>
#include <string>

namespace o2 {
namespace framework {

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

  template <class T>
  T get(const char *key) const {
    throw std::runtime_error("parameter type not implemented");
  };
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

template <> inline std::string ConfigParamRegistry::get<std::string>(const char *key) const {
  assert(mRetriever.get());
  return mRetriever->getString(key);
}

template <> inline bool ConfigParamRegistry::get<bool>(const char *key) const {
  assert(mRetriever.get());
  return mRetriever->getBool(key);
}

}
}

#endif //FRAMEWORK_CONFIGPARAMREGISTRY_H
