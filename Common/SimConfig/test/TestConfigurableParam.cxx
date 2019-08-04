// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <SimConfig/ConfigurableParam.h>
#include <SimConfig/ConfigurableParamHelper.h>
#include <SimConfig/SimConfig.h>
#include <iostream>
#include <TInterpreter.h>

class BazParam : public o2::conf::ConfigurableParamHelper<BazParam>
{
 public:
  double getGasDensity() const { return mGasDensity; }

 private:
  double mGasDensity = 2.0;
  std::array<int, 3> mPos = {1, 2, 3};

  O2ParamDef(BazParam, "Baz");
};
O2ParamImpl(BazParam);

int main(int argc, char* argv[])
{
  // generate dictionary for BazParam (done here since this is just
  // a tmp test class)
  gInterpreter->GenerateDictionary("BazParam");
  gInterpreter->GenerateTClass("BazParam", false);

  auto& conf = o2::conf::SimConfig::Instance();
  conf.resetFromArguments(argc, argv);

  // prints all parameters from any ConfigurableParam
  o2::conf::ConfigurableParam::printAllKeyValuePairs();

  // writes a configuration file
  o2::conf::ConfigurableParam::writeINI("initialconf.ini");

  // override some keys from command line
  o2::conf::ConfigurableParam::updateFromString(conf.getKeyValueString());

  // query using C++ object + API
  auto d1 = BazParam::Instance().getGasDensity();

  // query from global parameter registry AND by string name
  auto d2 = o2::conf::ConfigurableParam::getValueAs<double>("Baz.mGasDensity");
  assert(d1 == d2);

  // update
  double x = 102;
  o2::conf::ConfigurableParam::setValue<double>("Baz", "mGasDensity", x);

  // check that update correctly synced
  auto d3 = o2::conf::ConfigurableParam::getValueAs<double>("Baz.mGasDensity");
  assert(d3 == BazParam::Instance().getGasDensity());
  assert(x == BazParam::Instance().getGasDensity());

  o2::conf::ConfigurableParam::writeINI("newconf.ini");
}
