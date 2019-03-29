// configures a TPythia6 class
//   usage: o2sim -g extgen --extGenFile pythia6.C
// options:                 --extGenFunc pythia6(14000., "pythia.settings")

/// \author R+Preghenella - October 2018

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TPythia6.h"
#include "FairGenerator.h"
#include "Generators/GeneratorTGenerator.h"
#include <fstream>
#include <iostream>
#endif

R__LOAD_LIBRARY(libpythia6)

void configure(TPythia6* py6, const char* params);

FairGenerator*
  pythia6(double energy = 14000., const char* params = nullptr)
{
  // instance and configure Pythia6
  auto py6 = TPythia6::Instance();
  if (params)
    configure(py6, params);
  py6->Initialize("CMS", "p", "p", energy);

  // instance and configure TGenerator interface
  auto tgen = new o2::eventgen::GeneratorTGenerator();
  tgen->setMomentumUnit(1.);        // [GeV/c]
  tgen->setEnergyUnit(1.);          // [GeV/c]
  tgen->setPositionUnit(0.1);       // [cm]
  tgen->setTimeUnit(3.3356410e-12); // [s]
  tgen->setTGenerator(py6);
  return tgen;
}

void configure(TPythia6* py6, const char* params)
{
  std::ifstream file(params);
  if (!file.is_open()) {
    std::cerr << "Cannot open configuration file: " << params << std::endl;
    return;
  };
  std::string line, command;
  while (std::getline(file, line)) {
    /** remove comments **/
    command = line.substr(0, line.find_first_of("#"));
    if (command.length() == 0)
      continue;
    py6->Pygive(command.c_str());
  }
  file.close();
}
