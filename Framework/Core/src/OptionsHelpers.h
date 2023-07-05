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
#ifndef O2_FRAMEWORK_OPTIONSHELPERS_H_
#define O2_FRAMEWORK_OPTIONSHELPERS_H_

#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#include <boost/program_options/variables_map.hpp>
#include <iosfwd>

namespace boost::program_options
{
class options_description;
class variables_map;
} // namespace boost::program_options

namespace o2::framework
{
struct OptionsHelpers {
  template <typename T>
  static T as(boost::program_options::variable_value const& v)
  {
    std::istringstream is(v.as<std::string>());
    T t;
    is >> t;
    return t;
  }

  static boost::program_options::options_description makeUniqueOptions(boost::program_options::options_description const& od);
};
} // namespace o2::framework
#endif // O2_FRAMEWORK_OPTIONSHELPERS_H_
