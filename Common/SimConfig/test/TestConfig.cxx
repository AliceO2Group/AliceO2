// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <SimConfig/SimConfig.h>
#include <iostream>
#include <TFile.h>
#include <TClass.h>

int main(int argc, char* argv[])
{
  auto& conf = o2::conf::SimConfig::Instance();
  conf.resetFromArguments(argc, argv);

  std::cout << "Selected VMC engine " << conf.getMCEngine() << "\n";
  std::cout << "Selected Modules:\n";

  auto v = conf.getActiveDetectors();
  for (auto& m : v) {
    std::cout << "@ " << m << "\n";
  }

  std::cout << "Selected Generator " << conf.getGenerator() << "\n";

  // serialize simconfig and read it back
  TFile outfile("simconfigout_test.root", "RECREATE");
  auto& configdata = conf.getConfigData();
  auto cl = TClass::GetClass(typeid(configdata));
  outfile.WriteObjectAny(&configdata, cl, "SimConfigData");
  outfile.Close();

  o2::conf::SimConfigData* inconfigdata = nullptr;
  TFile file("simconfigout_test.root", "OPEN");
  file.GetObject("SimConfigData", inconfigdata);
  file.Close();

  if (inconfigdata) {
    conf.resetFromConfigData(*inconfigdata);
    auto v2 = conf.getActiveDetectors();
    for (auto& m : v2) {
      std::cout << "@ " << m << "\n";
    }
    return 0;
  } else {
    return 1;
  }
  return 0;
}
