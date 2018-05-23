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

#include "Framework/ParamRetriever.h"

#include <string>
#include <vector>

class FairMQProgOptions;

namespace o2
{
namespace framework
{

class FairOptionsRetriever : public ParamRetriever {
public:
  FairOptionsRetriever(const FairMQProgOptions *opts);

  int getInt(const char *name) const final;
  float getFloat(const char *name) const final;
  double getDouble(const char *name) const final;
  bool getBool(const char *name) const final;
  std::string getString(const char *name) const final;
  std::vector<std::string> getVString(const char *name) const final;
private:
  const FairMQProgOptions *mOpts;
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_BOOSTOPTIONSRETRIEVER_H

