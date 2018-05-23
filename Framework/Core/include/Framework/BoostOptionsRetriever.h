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

#include <boost/program_options.hpp>
#include <string>
#include <vector>

namespace o2
{
namespace framework
{

// FIXME: For the moment we only support basic types. Should probably support
//        (nested) vectors and maps as well. If FairMQ plugins can be used
//        without a device, this should probably be dropped in favor of
//        using the FairMQ plugin directly
class BoostOptionsRetriever : public ParamRetriever {
public:
  BoostOptionsRetriever(std::vector<ConfigParamSpec> &specs, bool ignoreUnknown, int &argc, char **&argv);

  virtual int getInt(const char *name) const final;
  virtual float getFloat(const char *name) const final;
  virtual double getDouble(const char *name) const final;
  virtual bool getBool(const char *name) const final;
  virtual std::string getString(const char *name) const final;
  virtual std::vector<std::string> getVString(const char *name) const final;
private:
  void parseArgs(int &argc, char **&argv);
  boost::program_options::variables_map mVariables;
  boost::program_options::options_description mDescription;
  bool mIgnoreUnknown;
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_BOOSTOPTIONSRETRIEVER_H

