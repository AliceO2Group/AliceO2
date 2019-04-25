// template macro to configure a generic TGenerator interface

/// \author R+Preghenella - March 2019

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TGenerator.h"
#include "FairGenerator.h"
#include "Generators/GeneratorTGenerator.h"
#endif

FairGenerator*
  tgenerator()
{
  // instance and configure an external TGenerator
  auto gen = new TGenerator;

  // instance and configure TGenerator interface
  auto tgen = new o2::eventgen::GeneratorTGenerator();
  tgen->setMomentumUnit(1.);        // [GeV/c]
  tgen->setEnergyUnit(1.);          // [GeV/c]
  tgen->setPositionUnit(0.1);       // [cm]
  tgen->setTimeUnit(3.3356410e-12); // [s]
  tgen->setTGenerator(gen);
  return tgen;
}
