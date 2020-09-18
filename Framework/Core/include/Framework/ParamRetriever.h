// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_PARAMRETRIEVER_H_
#define O2_FRAMEWORK_PARAMRETRIEVER_H_

#include "Framework/ConfigParamSpec.h"

#include <boost/property_tree/ptree_fwd.hpp>
#include <string>
#include <vector>

namespace o2::framework
{
/// Base class for extracting Configuration options from a given backend (e.g.
/// command line options).
class ParamRetriever
{
 public:
  virtual void update(std::vector<ConfigParamSpec> const& specs,
                      boost::property_tree::ptree& store,
                      boost::property_tree::ptree& provenance) = 0;
  virtual ~ParamRetriever() = default;
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_PARAMRETRIEVER_H_
