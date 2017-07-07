#if (!defined(__CINT__) && !defined(__CLING__)) || defined(__MAKECINT__)
#include "DetectorsPassive/Cave.h"
#include "DetectorsPassive/FrameStructure.h"
#include "FairRunSim.h"
#include "TGeoManager.h"
#include "TOFSimulation/Detector.h"
#include "TROOT.h"
#include "TString.h"
#include "TString.h"
#include "TSystem.h"

#include <boost/program_options.hpp>
#include <iomanip>
#include <iostream>
#endif

void drawTOF()
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

  o2::Passive::FrameStructure* frame = new o2::Passive::FrameStructure("Frame", "Frame");
  run->AddModule(frame);

  o2::tof::Detector* tof = new o2::tof::Detector("TOF", kTRUE);
  run->AddModule(tof);

  run->Init();
  {
    const TString ToHide =
      "cave B077 BREF1 B076 BIH142 BIH242 BIV42 B033 B034 B080 B081 BREF2 B047 B048 BM49 B049 B050 B051 B052 B045 B046 "
      "BSEGMO13 BSEGMO14 BSEGMO15 BSEGMO16 BSEGMO17 BSEGMO0 BSEGMO1 BSEGMO2 BSEGMO3 BSEGMO4 BSEGMO5 BSEGMO6 BSEGMO7 "
      "BSEGMO8 BSEGMO9 BSEGMO10 BSEGMO11 BSEGMO12 B072 B073 BIH172 BIH272 BIV72 B063 B063A B063I B063IA B163 B163A "
      "B163I B163IA B263 B263A B263I B263IA B363 B363A B363I B363IA B463 B463A B463I B463IA BA59 BA62 BTSH_M BTSHA_M "
      "BTSHT1_M BTSHT2_M BTSHT3_M BTSHT4_M BTSH_AM BTSHA_AM BTSHT1_AM BTSHT2_AM BTSHT3_AM BTSHT4_AM BTSH_A BTSHA_A "
      "BTSHT1_A BTSHT2_A BTSHT3_A BTSHT4_A BTRD0 BTRD1 BTRD2 BTRD3 BTRD4 BTRD5 BTRD6 BTRD7 BTRD8 BTRD9 BTRD10 BTRD11 "
      "BTRD12 BTRD13 BTRD14 BTRD15 BTRD16 BTRD17 BTOF0 BTOF1 BTOF2 BTOF3 BTOF4 BTOF5 BTOF6 BTOF7 BTOF8 BTOF9 BTOF10 "
      "BTOF11 BTOF12 BTOF13 BTOF14 BTOF15 BTOF16 BTOF17 BRS1 BRS2 BRS3 BRS4 BFMO BFTRD BFIR BFII BFOR BFOO BFLB BFLL "
      "BFRB BFRR BBMO BBCE BBTRD BBLB BBLL BBRB BBRR BBC1 BBC2 BBC3 BBC4 BBD1 BBD3 BBD2 BBD4 FTOA FTOB FTOC FLTA FLTB "
      "FLTC FWZ1D FWZAD FWZ1U FWZBU FWZ2 FWZC FWZ3 FWZ4 FSTR FHON FPC1 FPC2 FPCB FSEN FSEZ FPAD FRGL FGLF FPEA FPEB "
      "FALT FALB FPE1 FPE4 FPE2 FPE3 FIF1 FIF2 FIF3 FFC1 FFC2 FFC3 FCC1 FCC2 FCC3 FAIA FAIB FAIC FCA1 FCA2 FFEA FAL1 "
      "FRO1 FREE FBAR FBA1 FBA2 FAL2 FAL3 FRO2 FTUB FITU FTLN FLO1 FLO2 FLO3 FBAS FBS1 FBS2 FCAB FCAL FCBL FSAW FCBB "
      "FCOV FCOB FCOP FTOS";

    TObjArray* lToHide = ToHide.Tokenize(" ");
    TIter* iToHide = new TIter(lToHide);
    TObjString* name;
    while ((name = (TObjString*)iToHide->Next()))
      gGeoManager->GetVolume(name->GetName())->SetVisibility(kFALSE);

    TString ToShow =
      "BTOF0 BFMO BFIR BFOR BFLB BFRB BBMO BBCE BBLB BBRB FTOA FTOB FTOC FLTA FLTB FLTC FWZ1D FWZAD FWZ1U FWZBU FWZ2 "
      "FWZC FWZ3 FWZ4 FSTR FHON FPC1 FPC2 FPCB FSEN FSEZ FPAD FRGL FGLF FPEA FPEB FALT FALB FPE1 FPE4 FPE2 FPE3 FIF1 "
      "FIF2 FIF3 FFC1 FFC2 FFC3 FCC1 FCC2 FCC3 FAIA FAIB FAIC FCA1 FCA2 FFEA FAL1 FRO1 FREE FBAR FBA1 FBA2 FAL2 FAL3 "
      "FRO2 FTUB FITU FTLN FLO1 FLO2 FLO3 FBAS FBS1 FBS2 FCAB FCAL FCBL FSAW FCBB FCOV FCOB FCOP FTOS";
    // ToShow.ReplaceAll("FCOV", "");//Remove external cover but PHOS hole
    // ToShow.ReplaceAll("FLTA", "");//Remove internal cover but PHOS hole
    ToShow.ReplaceAll("FFC1", ""); // Remove internal cover but PHOS hole
    ToShow.ReplaceAll("FFC2", ""); // Remove internal cover but PHOS hole
    ToShow.ReplaceAll("FFC3", ""); // Remove internal cover but PHOS hole
    ToShow.ReplaceAll("FALT", ""); // Remove internal cover but PHOS hole
    ToShow.ReplaceAll("FALB", ""); // Remove internal cover but PHOS hole
    ToShow.ReplaceAll("FIF1", ""); // Remove internal cover but PHOS hole
    ToShow.ReplaceAll("FIF2", ""); // Remove internal cover but PHOS hole
    ToShow.ReplaceAll("FIF3", ""); // Remove internal cover but PHOS hole
    ToShow.ReplaceAll("FCOP", ""); // Remove internal cover but PHOS hole

    TObjArray* lToShow = ToShow.Tokenize(" ");
    TIter* iToShow = new TIter(lToShow);
    while ((name = (TObjString*)iToShow->Next()))
      gGeoManager->GetVolume(name->GetName())->SetVisibility(kTRUE);

    const TString ToTrans = "FTOS FCOV FLTA";

    TObjArray* lToTrans = ToTrans.Tokenize(" ");
    TIter* iToTrans = new TIter(lToTrans);
    while ((name = (TObjString*)iToTrans->Next()))
      gGeoManager->GetVolume(name->GetName())->SetTransparency(50);
  }

  gGeoManager->GetListOfVolumes()->ls();
  gGeoManager->CloseGeometry();

  gGeoManager->GetTopVolume()->Draw("ogl");
  gGeoManager->Export("TOFgeometry.root");
}
