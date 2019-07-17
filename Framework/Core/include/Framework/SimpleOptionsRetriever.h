// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_SIMPLEOPTIONSRETRIEVER_H_
#define O2_FRAMEWORK_SIMPLEOPTIONSRETRIEVER_H_

#include "Framework/ParamRetriever.h"

namespace o2::framework
{

// Simple standalone param retriever to be populated programmatically or via a
// predefined ptree.
class SimpleOptionsRetriever : public ParamRetriever
{
 public:
  int getInt(const char* name) const final;
  float getFloat(const char* name) const final;
  double getDouble(const char* name) const final;
  bool getBool(const char* name) const final;
  std::string getString(const char* name) const final;
  boost::property_tree::ptree getPTree(const char* name) const final;

  void setInt(char const* name, int);
  void setFloat(char const* name, float);
  void setDouble(char const* name, double);
  void setBool(char const* name, bool);
  void setString(char const* name, std::string const&);
  void setPTree(char const* name, boost::property_tree::ptree const& tree);

 private:
  boost::property_tree::ptree mStore;
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_SIMPLEOPTIONSRETRIEVER_H_
