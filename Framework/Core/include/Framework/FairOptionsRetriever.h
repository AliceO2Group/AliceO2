// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_FAIROPTIONSRETRIEVER_H_
#define O2_FRAMEWORK_FAIROPTIONSRETRIEVER_H_

#include "Framework/ParamRetriever.h"
#include "Framework/ConfigParamSpec.h"

#include <boost/property_tree/ptree_fwd.hpp>
#include <fairmq/ProgOptionsFwd.h>

#include <vector>

namespace o2::framework
{

class FairOptionsRetriever : public ParamRetriever
{
 public:
  FairOptionsRetriever(const FairMQProgOptions* opts) : mOpts{opts} {}

  void update(std::vector<ConfigParamSpec> const& schema,
              boost::property_tree::ptree& store,
              boost::property_tree::ptree& provenance) override;

 private:
  const FairMQProgOptions* mOpts;
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_FAIROPTIONSRETRIEVER_H_
