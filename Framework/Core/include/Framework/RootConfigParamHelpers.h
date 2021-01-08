// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_ROOTCONFIGPARAMHELPERS_H_
#define O2_FRAMEWORK_ROOTCONFIGPARAMHELPERS_H_

#include "Framework/ConfigParamSpec.h"
#include "Framework/RuntimeError.h"
#include <TClass.h>
#include <boost/property_tree/ptree_fwd.hpp>
#include <type_traits>
#include <typeinfo>

namespace o2::framework
{

/// Helpers to Serialise / Deserialise ROOT objects using the ConfigParamSpec mechanism
struct RootConfigParamHelpers {
  static std::vector<ConfigParamSpec> asConfigParamSpecsImpl(std::string const& mainkey, TClass* cl, void* obj);
  /// Given a TClass, fill the object in obj as if it was member of the former,
  /// using the values in the ptree to override, where appropriate.
  static void fillFromPtree(TClass* cl, void* obj, boost::property_tree::ptree const& pt);

  // Grab the list of data members of a type T and construct the list of
  // associated ConfigParamSpec. Optionally provide a prototype object @a proto
  // to use for the defaults.
  template <typename T>
  static std::vector<ConfigParamSpec> asConfigParamSpecs(std::string const& mainKey, T const& proto = T{})
  {
    auto cl = TClass::GetClass<T>();
    if (!cl) {
      throw runtime_error_f("Unable to convert object %s", typeid(T).name());
    }

    return asConfigParamSpecsImpl(mainKey, cl, reinterpret_cast<void*>(const_cast<T*>(&proto)));
  }

  /// Given a ptree use it to create a (ROOT serialised) object of type T,
  /// where the default values of the object are overriden by those passed
  /// in the ptree.
  template <typename T>
  static T as(boost::property_tree::ptree const& pt)
  {
    T obj;
    TClass* cl = TClass::GetClass<T>();
    fillFromPtree(cl, reinterpret_cast<void*>(&obj), pt);
    return obj;
  }
};

} // namespace o2::framework

#endif /* O2_FRAMEWORK_ROOTCONFIGPARAMHELPERS_H_ */
