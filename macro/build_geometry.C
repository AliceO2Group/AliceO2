// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TGeoManager.h"
#include "TString.h"
#include "TSystem.h"

#include <DetectorsBase/Detector.h>
#include <Field/MagneticField.h>
#include <SimConfig/SimConfig.h>
#include "FairRunSim.h"
#include <FairLogger.h>
#include <algorithm>
#endif

void finalize_geometry(FairRunSim* run);

// decides whether the FairModule named 's' has been
// requested in the configuration (or is required
// for some reason -cavern or frame-)
bool isActivated(std::string s)
{
  if (s == "CAVE") {
    // we always need the cavern
    return true;
  }

  // access user configuration for list of wanted modules
  const auto& modulelist = o2::conf::SimConfig::Instance().getActiveDetectors();
  auto active = (std::find(modulelist.begin(), modulelist.end(), s) != modulelist.end());

  if (s == "FRAME") {
    // the frame structure must be present to support other detectors
    return active || isActivated("TOF") || isActivated("TRD");
  }

  return active;
}

// create a number of FairModules, either active or passive
void createModules(FairRunSim& runner, const std::vector<std::string>& modules, bool isActive)
{
  for (auto& moduleName : modules) {
    if (isActivated(moduleName)) {
      auto module = o2::Base::createFairModule(moduleName.c_str(), isActive);
      if (!module) {
        LOG(ERROR) << "Could not create module " << moduleName << "\n";
        throw;
      }
      runner.AddModule(module);
    }
  }
}

// a "factory" like macro to instantiate the O2 geometry
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

  bool isActive{ true };

  createModules(*run, { "CAVE", "ABSO", "DIPO", "FRAME", "HALL", "MAG", "PIPE", "SHIL" }, !isActive);

  createModules(*run, { "EMC", "FIT", "ITS", "MFT", "PHS", "TOF", "TPC", "TRD" }, isActive);

  if (geomonly) {
    run->Init();
    finalize_geometry(run);
    gGeoManager->Export("O2geometry.root");
  }
}

void finalize_geometry(FairRunSim* run)
{
  // finalize geometry and declare alignable volumes
  // this should be called geometry is fully built

  if (!gGeoManager) {
    LOG(ERROR) << "gGeomManager is not available" << FairLogger::endl;
    return;
  }

  gGeoManager->CloseGeometry();
  if (!run) {
    LOG(ERROR) << "FairRunSim is not available" << FairLogger::endl;
    return;
  }

  const TObjArray* modArr = run->GetListOfModules();
  TIter next(modArr);
  FairModule* module = nullptr;
  while ((module = (FairModule*)next())) {
    o2::Base::Detector* det = dynamic_cast<o2::Base::Detector*>(module);
    if (det)
      det->addAlignableVolumes();
  }
}
