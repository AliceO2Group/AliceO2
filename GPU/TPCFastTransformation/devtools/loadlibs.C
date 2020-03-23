/*
   works only with ROOT >= 6

   alienv load ROOT/latest-root6
   alienv load Vc/latest
   root -l
   .x loadlibs.C
 */

#include "TSystem.h"
#include "TROOT.h"

{
  gSystem->AddIncludePath("-I../. -I$VC_ROOT/include");
  gSystem->AddLinkedLibs("$VC_ROOT/lib/libVc.a");
  gROOT->LoadMacro("../IrregularSpline1D.cxx++");
  gROOT->LoadMacro("../IrregularSpline2D3D.cxx++");
}
