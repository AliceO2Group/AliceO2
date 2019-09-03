#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TString.h"
#include "FairRunAna.h"
#include "FairFileSource.h"
#include "FairEventManager.h"
#include "FairRuntimeDb.h"
#include "FairParRootFileIo.h"
#include "FairMCTracks.h"
#include "FairMCPointDraw.h"
#endif

void eventDisplay()
{
  //-----User Settings:-----------------------------------------------
  TString InputFile = "test.root";
  TString ParFile = "params.root";
  TString OutFile = "tst.root";

  // -----   Reconstruction run   -------------------------------------------
  FairRunAna* fRun = new FairRunAna();
  FairFileSource* fFileSource = new FairFileSource(InputFile.Data());
  fRun->SetSource(fFileSource);
  fRun->SetOutputFile(OutFile.Data());

  FairRuntimeDb* rtdb = fRun->GetRuntimeDb();
  FairParRootFileIo* parInput1 = new FairParRootFileIo();
  parInput1->open(ParFile.Data());
  rtdb->setFirstInput(parInput1);

  FairEventManager* fMan = new FairEventManager();

  //----------------------Traks and points -------------------------------------
  FairMCTracks* Track = new FairMCTracks("Monte-Carlo Tracks");
  FairMCPointDraw* TorinoDetectorPoints = new FairMCPointDraw("FairTestDetectorPoint", kRed, kFullSquare);

  fMan->AddTask(Track);
  fMan->AddTask(TorinoDetectorPoints);

  fMan->Init();
}
