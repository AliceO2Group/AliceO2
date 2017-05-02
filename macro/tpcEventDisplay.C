//#if (!defined(__CINT__) && !defined(__CLING__)) || defined(__MAKECINT__)
#include "TString.h"
#include "FairRunAna.h"
#include "FairFileSource.h"
#include "FairRuntimeDb.h"
#include "FairParRootFileIo.h"
#include "FairEventManager.h"
#include "FairMCTracks.h"
#include "TPCEventdisplay/HitDraw.h"
//#endif

using namespace o2::TPC;

void tpcEventDisplay(TString InputFile, TString ParFile)
{
  //-----User Settings:-----------------------------------------------
  //TString  InputFile     ="test.root";
  //TString  ParFile       ="params.root";
  TString  OutFile	 ="tst.root";


  // -----   Reconstruction run   -------------------------------------------
  FairRunAna *fRun= new FairRunAna();
  FairFileSource *fFileSource = new FairFileSource(InputFile.Data());
  fRun->SetSource(fFileSource);
  fRun->SetOutputFile(OutFile.Data());

  FairRuntimeDb* rtdb = fRun->GetRuntimeDb();
  FairParRootFileIo* parInput1 = new FairParRootFileIo();
  parInput1->open(ParFile.Data());
  rtdb->setFirstInput(parInput1);

  FairEventManager *fMan= new FairEventManager();

  //----------------------Traks and points -------------------------------------
  FairMCTracks    *Track     = new FairMCTracks("Monte-Carlo Tracks");
  FairPointSetDraw *pointSet = new HitDraw("TPCPoint", kRed, kFullSquare);

  fMan->AddTask(Track);
  fMan->AddTask(pointSet);


  fMan->Init();

}
