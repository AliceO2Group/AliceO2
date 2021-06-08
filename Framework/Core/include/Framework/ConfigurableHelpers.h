// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_CONFIGURABLEHELPERS_H_
#define O2_FRAMEWORK_CONFIGURABLEHELPERS_H_

#include "Framework/ConfigParamSpec.h"
#include "Framework/Configurable.h"
#include "Framework/RootConfigParamHelpers.h"

namespace o2::framework
{

struct ConfigurableHelpers {
  template <typename T, ConfigParamKind K, typename IP>
  static bool appendOption(std::vector<ConfigParamSpec>& options, Configurable<T, K, IP>& what)
  {
    if constexpr (variant_trait_v<typename std::decay<T>::type> != VariantType::Unknown) {
      options.emplace_back(ConfigParamSpec{what.name, variant_trait_v<std::decay_t<T>>, what.value, {what.help}, what.kind});
    } else {
      auto specs = RootConfigParamHelpers::asConfigParamSpecs<T>(what.name, what.value);
      options.insert(options.end(), specs.begin(), specs.end());
    }
    return true;
  }
};

} // namespace o2::framework
#endif //  O2_FRAMEWORK_CONFIGURABLEHELPERS_H_
