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
#ifndef O2_FRAMEWORK_SIMPLEOPTIONSRETRIEVER_H_
#define O2_FRAMEWORK_SIMPLEOPTIONSRETRIEVER_H_

#include "Framework/ParamRetriever.h"
#include <boost/property_tree/ptree.hpp>
#include <string>

namespace o2::framework
{

// Simple standalone param retriever to be populated programmatically or via a
// predefined ptree.
class SimpleOptionsRetriever : public ParamRetriever
{
 public:
  SimpleOptionsRetriever(boost::property_tree::ptree& tree, std::string const& provenanceLabel)
    : mTree{tree},
      mProvenanceLabel{provenanceLabel}
  {
  }

  void update(std::vector<ConfigParamSpec> const& specs,
              boost::property_tree::ptree& store,
              boost::property_tree::ptree& provenance) override;

 private:
  boost::property_tree::ptree mTree;
  std::string mProvenanceLabel;
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_SIMPLEOPTIONSRETRIEVER_H_
