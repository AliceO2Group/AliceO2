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

#include <boost/property_tree/ptree.hpp>
#include <string>
#include <vector>

namespace o2
{
namespace framework
{

/// Base class for extracting Configuration options from a given backend (e.g.
/// command line options).
class ParamRetriever
{
 public:
  virtual int getInt(const char* name) const = 0;
  virtual float getFloat(const char* name) const = 0;
  virtual double getDouble(const char* name) const = 0;
  virtual bool getBool(const char* name) const = 0;
  virtual std::string getString(const char* name) const = 0;
  virtual boost::property_tree::ptree getPTree(const char* name) const = 0;
  virtual ~ParamRetriever() = default;
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_PARAMETERRETRIEVER_H
