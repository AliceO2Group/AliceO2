#include <iostream>
#include "TSystem.h"
#include "TROOT.h"
#include "TString.h"

int main(int argc, char** argv)
{
  Int_t error;
  const char* macro = "$HOME/macro.C";

  gSystem->ListLibraries();
  // gROOT->ProcessLine(Form(".L %s",macro), &error);
  // std::cout << "Exit code from loading=" << error << "\n";
  // if (error) {
  //     std::cout << "Could not load macro" << macro << "\n";
  //     return error;
  // }
  error = gSystem->CompileMacro(macro,"c");

  std::cout << "Exit code from compilation=" << error << "\n";

  return !error;
}
