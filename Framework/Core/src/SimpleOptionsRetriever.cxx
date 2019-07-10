// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/SimpleOptionsRetriever.h"
#include "Framework/ConfigParamSpec.h"

#include "PropertyTreeHelpers.h"

#include <boost/program_options.hpp>

#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>

using namespace o2::framework;
namespace bpo = boost::program_options;

namespace o2::framework
{

int SimpleOptionsRetriever::getInt(char const* key) const
{
  return mStore.get<int>(key);
}

float SimpleOptionsRetriever::getFloat(char const* key) const
{
  return mStore.get<float>(key);
}

double SimpleOptionsRetriever::getDouble(char const* key) const
{
  return mStore.get<double>(key);
}

bool SimpleOptionsRetriever::getBool(char const* key) const
{
  return mStore.get<bool>(key);
}

std::string SimpleOptionsRetriever::getString(char const* key) const
{
  return mStore.get<std::string>(key);
}

boost::property_tree::ptree SimpleOptionsRetriever::getPTree(char const* key) const
{
  return mStore.get_child(key);
}

void SimpleOptionsRetriever::setInt(char const* key, int value)
{
  mStore.put<int>(key, value);
}

void SimpleOptionsRetriever::setFloat(char const* key, float value)
{
  mStore.put<float>(key, value);
}

void SimpleOptionsRetriever::setDouble(char const* key, double value)
{
  mStore.put<double>(key, value);
}

void SimpleOptionsRetriever::setBool(char const* key, bool value)
{
  mStore.put<bool>(key, value);
}

void SimpleOptionsRetriever::setString(char const* key, std::string const& value)
{
  mStore.put<std::string>(key, value);
}

void SimpleOptionsRetriever::setPTree(char const* key, boost::property_tree::ptree const& value)
{
  mStore.put_child(key, value);
}

} // namespace o2::framework
