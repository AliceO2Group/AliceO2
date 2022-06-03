#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TGeoManager.h"
#include "TString.h"
#include "TSystem.h"

#include "FairRunSim.h"
#include <FairRootFileSink.h>
#include "DetectorsPassive/Cave.h"
#include "MIDSimulation/Detector.h"

#endif

// Macro to export the MID geometry into a ROOT file for vizualisation
void drawMIDGeometry()
{

  TString dir = getenv("VMCWORKDIR");
  TString geom_dir = dir + "/Detectors/Geometry/";
  gSystem->Setenv("GEOMPATH", geom_dir.Data());

  TString tut_configdir = dir + "/Detectors/gconfig";
  gSystem->Setenv("CONFIG_DIR", tut_configdir.Data());

  // Create simulation run
  FairRunSim* run = new FairRunSim();
  run->SetSink(new FairRootFileSink("foo.root")); // Output file
  run->SetName("TGeant3");        // Transport engine
  // Create media
  run->SetMaterials("media.geo"); // Materials

  // Create geometry
  o2::passive::Cave* cave = new o2::passive::Cave("CAVE");
  cave->SetGeometryFileName("cave.geo");
  run->AddModule(cave);

  o2::mid::Detector* mid = new o2::mid::Detector(kTRUE);
  run->AddModule(mid);

  run->Init();

  gGeoManager->SetVisLevel(5);
  gGeoManager->Export("MIDgeometry.root");
}
