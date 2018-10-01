// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//first version 8/2018, Sandro Wenzel

#ifndef COMMON_SIMCONFIG_INCLUDE_SIMCONFIG_CONFIGURABLEPARAMHELPER_H_
#define COMMON_SIMCONFIG_INCLUDE_SIMCONFIG_CONFIGURABLEPARAMHELPER_H_

#include "SimConfig/ConfigurableParam.h"
#include "TClass.h"
#include <type_traits>
#include <typeinfo>

namespace o2
{
namespace conf
{

// just a (non-templated) Helper with exclusively private functions
// used by ConfigurableParamHelper
class _ParamHelper
{
 private:
  static void printParametersImpl(TClass* cl, void*);

  static void fillKeyValuesImpl(std::string mainkey, TClass* cl, void*, boost::property_tree::ptree*,
                                std::map<std::string, std::pair<int, void*>>*);

  static void printWarning(std::type_info const&);

  template <typename P>
  friend class ConfigurableParamHelper;
};

// implementer (and checker) for concrete ConfigurableParam classes P
template <typename P>
class ConfigurableParamHelper : virtual public ConfigurableParam
{
 public:
  using ConfigurableParam::ConfigurableParam;
  static P& Instance()
  {
    return P::sInstance;
  }

  std::string getName() final
  {
    return P::sKey;
  }

  // one of the key methods, using introspection to print itself
  void printKeyValues() final
  {
    // just a helper line to make sure P::sInstance is looked-up
    // and that compiler complains about missing static sInstance of type P
    // volatile void* ptr = (void*)&P::sInstance;
    // static assert on type of sInstance:
    static_assert(std::is_same<decltype(P::sInstance), P>::value, "static instance must of same type as class");

    // obtain the TClass for P and delegate further
    auto cl = TClass::GetClass(typeid(P));
    if (!cl) {
      _ParamHelper::printWarning(typeid(P));
      return;
    }
    _ParamHelper::printParametersImpl(cl, (void*)this);
  }

  void putKeyValues(boost::property_tree::ptree* tree) final
  {
    auto cl = TClass::GetClass(typeid(P));
    if (!cl) {
      _ParamHelper::printWarning(typeid(P));
      return;
    }
    _ParamHelper::fillKeyValuesImpl(getName(), cl, (void*)this, tree, sKeyToStorageMap);
  }
};
} // namespace conf
} // namespace o2

#endif /* COMMON_SIMCONFIG_INCLUDE_SIMCONFIG_CONFIGURABLEPARAMHELPER_H_ */
