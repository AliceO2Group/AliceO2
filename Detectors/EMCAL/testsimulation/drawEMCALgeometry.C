#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "DetectorsPassive/Cave.h"
#include "DetectorsPassive/FrameStructure.h"
#include "EMCALSimulation/Detector.h"
#include "FairRunSim.h"
#include "TGeoManager.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TROOT.h"
#include "TString.h"
#include "TSystem.h"

#include <boost/program_options.hpp>
#include <iomanip>
#include <iostream>
#endif

void drawEMCALgeometry()
{
  // minimal macro to test setup of the geometry

  TString dir = getenv("VMCWORKDIR");
  TString geom_dir = dir + "/Detectors/Geometry/";
  gSystem->Setenv("GEOMPATH", geom_dir.Data());

  TString tut_configdir = dir + "/Detectors/gconfig";
  gSystem->Setenv("CONFIG_DIR", tut_configdir.Data());

  // Create simulation run
  FairRunSim* run = new FairRunSim();
  run->SetOutputFile("foo.root"); // Output file
  run->SetName("TGeant3");        // Transport engine
  // Create media
  run->SetMaterials("media.geo"); // Materials

  // Create geometry

  o2::Passive::Cave* cave = new o2::Passive::Cave("CAVE");
  cave->SetGeometryFileName("cave.geo");
  run->AddModule(cave);

  o2::passive::FrameStructure* frame = new o2::passive::FrameStructure("Frame", "Frame");
  run->AddModule(frame);

  o2::EMCAL::Detector* emcal = new o2::EMCAL::Detector(kTRUE);
  run->AddModule(emcal);

  run->Init();
  {
    const TString ToHide = "cave";

    TObjArray* lToHide = ToHide.Tokenize(" ");
    TIter* iToHide = new TIter(lToHide);
    TObjString* name;
    while ((name = (TObjString*)iToHide->Next()))
      gGeoManager->GetVolume(name->GetName())->SetVisibility(kFALSE);

    TString ToShow = "SMOD SM3rd DCSM DCEXT";
    // ToShow.ReplaceAll("FCOV", "");//Remove external cover but PHOS hole
    // ToShow.ReplaceAll("FLTA", "");//Remove internal cover but PHOS hole

    TObjArray* lToShow = ToShow.Tokenize(" ");
    TIter* iToShow = new TIter(lToShow);
    while ((name = (TObjString*)iToShow->Next()))
      gGeoManager->GetVolume(name->GetName())->SetVisibility(kTRUE);

    const TString ToTrans = "SCM0 SCMX SCMY";

    TObjArray* lToTrans = ToTrans.Tokenize(" ");
    TIter* iToTrans = new TIter(lToTrans);
    while ((name = (TObjString*)iToTrans->Next())) {
      auto v = gGeoManager->GetVolume(name->GetName());
      if (v)
        v->SetTransparency(50);
      else
        printf("Volume %s not found ...\n", name->GetName());
    }
  }

  gGeoManager->GetListOfVolumes()->ls();
  gGeoManager->CloseGeometry();

  gGeoManager->GetTopVolume()->Draw("ogl");
  gGeoManager->Export("EMCALgeometry.root");
}
