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
#ifndef O2_COMMON_UTILS_CONFIGURABLEPARAMREADERS_H_
#define O2_COMMON_UTILS_CONFIGURABLEPARAMREADERS_H_

#include <boost/property_tree/ptree.hpp>
#include <string>

namespace o2::conf
{

// Helpers to read ConfigurableParam from different file formats
class ConfigurableParamReaders
{
 public:
  static void setInputDir(const std::string& d) { sInputDir = d; }
  static const std::string& getInputDir() { return sInputDir; }

  static boost::property_tree::ptree readINI(std::string const& filepath);
  static boost::property_tree::ptree readJSON(std::string const& filepath);
  static boost::property_tree::ptree readConfigFile(std::string const& filepath);

 private:
  static std::string sInputDir;
};

} // namespace o2::conf
#endif // O2_COMMON_UTILS_CONF_CONFIGURABLEPARAMREADERS_H_
