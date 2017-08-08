#if (!defined(__CINT__) && !defined(__CLING__)) || defined(__MAKECINT__)
#include "TGeoManager.h"
#include "TString.h"
#include "TSystem.h"

#include "DetectorsPassive/Cave.h"
#include "DetectorsPassive/FrameStructure.h"
#include "FairRunSim.h"
#endif

// macro to test correct initialization of the
// (complete) geometry structure
void build_geometry(FairRunSim* run = nullptr)
{
  bool geomonly = (run == nullptr);

  // minimal macro to test setup of the geometry

  TString dir = getenv("VMCWORKDIR");
  TString geom_dir = dir + "/Detectors/Geometry/";
  gSystem->Setenv("GEOMPATH", geom_dir.Data());

  TString tut_configdir = dir + "/Detectors/gconfig";
  gSystem->Setenv("CONFIG_DIR", tut_configdir.Data());

  // Create simulation run if it does not exist
  if (run == nullptr) {
    run = new FairRunSim();
    run->SetOutputFile("foo.root"); // Output file
    run->SetName("TGeant3");        // Transport engine
  }
  // Create media
  run->SetMaterials("media.geo"); // Materials

  // we need a field to properly init the media
  auto field = new o2::field::MagneticField("Maps", "Maps", -1., -1., o2::field::MagFieldParam::k5kG);
  run->SetField(field);

  // Create geometry
  o2::Passive::Cave* cave = new o2::Passive::Cave("CAVE");
  // the experiment hall (cave)
  cave->SetGeometryFileName("cave.geo");
  run->AddModule(cave);

  // the frame structure to support other detectors
  auto frame = new o2::passive::FrameStructure("Frame", "Frame");
  run->AddModule(frame);

  // TOF
  auto tof = new o2::tof::Detector("TOF", true);
  run->AddModule(tof);

  // TRD
  auto trd = new o2::trd::Detector("TRD", true);
  run->AddModule(trd);

  // tpc
  auto tpc = new o2::TPC::Detector("TPC", true);
  run->AddModule(tpc);

  // emcal
  run->AddModule(new o2::EMCAL::Detector("EMCAL", true));

  if (geomonly) {
    run->Init();
    gGeoManager->Export("O2geometry.root");
  }
}
