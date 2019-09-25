#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <iostream>
#include "TSystem.h"
#endif

void load_all_libs()
{
  gSystem->Load("libDetectorsBase");
  gSystem->Load("libDetectorsPassive");
  gSystem->Load("libField");
  gSystem->Load("libGenerators");
  gSystem->Load("libHeaders");
  gSystem->Load("libHitAnalysis");
  gSystem->Load("libITSBase");
  gSystem->Load("libITSReconstruction");
  gSystem->Load("libITSSimulation");
  gSystem->Load("libMFTBase");
  gSystem->Load("libMFTReconstruction");
  gSystem->Load("libMFTSimulation");
  gSystem->Load("libMathUtils");
  gSystem->Load("libO2Device");
  gSystem->Load("libSimulationDataFormat");
  gSystem->Load("libTPCBase");
  gSystem->Load("libTPCSimulation");
  gSystem->Load("libaliceHLTwrapper");
  cout << endl
       << endl;
  cout << "Macro finished succesfully." << endl;
}
