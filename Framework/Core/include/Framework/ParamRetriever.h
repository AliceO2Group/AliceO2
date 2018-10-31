// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_PARAMETERRETRIEVER_H
#define FRAMEWORK_PARAMETERRETRIEVER_H

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
class ParamRetriever {
public:
  virtual int getInt(const char *name) const = 0;
  virtual float getFloat(const char *name) const = 0;
  virtual double getDouble(const char *name) const = 0;
  virtual bool getBool(const char *name) const = 0;
  virtual std::string getString(const char *name) const = 0;
  virtual std::vector<std::string> getVString(const char *name) const = 0;
  virtual ~ParamRetriever() = default;
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_PARAMETERRETRIEVER_H
