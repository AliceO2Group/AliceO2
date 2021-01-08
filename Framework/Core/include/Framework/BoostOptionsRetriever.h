// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_BOOSTOPTIONSRETRIEVER_H
#define FRAMEWORK_BOOSTOPTIONSRETRIEVER_H

#include "Framework/ConfigParamSpec.h"
#include "Framework/ParamRetriever.h"

#include <boost/property_tree/ptree_fwd.hpp>
#include <memory>
#include <vector>

namespace boost
{
namespace program_options
{
class options_description;
}
} // namespace boost

namespace o2::framework
{

/// This extracts the specified ConfigParams from (argc, argv) and makes them
/// available to the ConfigParamRegistry.
class BoostOptionsRetriever : public ParamRetriever
{
 public:
  BoostOptionsRetriever(bool ignoreUnknown,
                        int argc, char** argv);
  void update(std::vector<ConfigParamSpec> const& specs,
              boost::property_tree::ptree& store,
              boost::property_tree::ptree& provenance) override;

 private:
  std::unique_ptr<boost::program_options::options_description> mDescription;
  int mArgc;
  char** mArgv;
  bool mIgnoreUnknown;
};

} // namespace o2::framework
#endif // FRAMEWORK_BOOSTOPTIONSRETRIEVER_H
