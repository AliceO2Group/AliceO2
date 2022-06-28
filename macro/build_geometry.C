// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TGeoManager.h"
#include "TString.h"
#include "TSystem.h"

#include "DetectorsPassive/Cave.h"
#include "DetectorsPassive/Magnet.h"
#include "DetectorsPassive/Dipole.h"
#include "DetectorsPassive/Compensator.h"
#include "DetectorsPassive/Absorber.h"
#include "DetectorsPassive/Shil.h"
#include "DetectorsPassive/Hall.h"
#include "DetectorsPassive/Pipe.h"
#include <Field/MagneticField.h>
#include <TPCSimulation/Detector.h>
#include <ITSSimulation/Detector.h>
#include <MFTSimulation/Detector.h>
#include <MCHSimulation/Detector.h>
#include <MIDSimulation/Detector.h>
#include <EMCALSimulation/Detector.h>
#include <TOFSimulation/Detector.h>
#include <TRDSimulation/Detector.h>
#include <FT0Simulation/Detector.h>
#include <FV0Simulation/Detector.h>
#include <FDDSimulation/Detector.h>
#include <HMPIDSimulation/Detector.h>
#include <PHOSSimulation/Detector.h>
#include <CPVSimulation/Detector.h>
#include <ZDCSimulation/Detector.h>
#include <DetectorsPassive/Cave.h>
#include <DetectorsPassive/FrameStructure.h>
#include <SimConfig/SimConfig.h>
#include <FairRunSim.h>
#include <FairRootFileSink.h>
#include <FairLogger.h>
#include <algorithm>
#include "DetectorsCommonDataFormats/UpgradesStatus.h"
#include <DetectorsBase/SimFieldUtils.h>
#endif

#ifdef ENABLE_UPGRADES
#include <ITS3Simulation/Detector.h>
#include <TRKSimulation/Detector.h>
#include <FT3Simulation/Detector.h>
#include <FCTSimulation/Detector.h>
#include <Alice3DetectorsPassive/Pipe.h>
#endif

void finalize_geometry(FairRunSim* run);

bool isActivated(std::string s)
{
  // access user configuration for list of wanted modules
  auto& modulelist = o2::conf::SimConfig::Instance().getActiveModules();
  auto active = std::find(modulelist.begin(), modulelist.end(), s) != modulelist.end();
  if (active) {
    LOG(info) << "Activating " << s << " module";
  }
  return active;
}

bool isReadout(std::string s)
{
  // access user configuration for list of wanted modules
  auto& modulelist = o2::conf::SimConfig::Instance().getReadoutDetectors();
  auto active = std::find(modulelist.begin(), modulelist.end(), s) != modulelist.end();
  if (active) {
    LOG(info) << "Reading out " << s << " detector";
  }
  return active;
}

// a "factory" like macro to instantiate the O2 geometry
void build_geometry(FairRunSim* run = nullptr)
{
  bool geomonly = (run == nullptr);

  // minimal macro to test setup of the geometry
  auto& confref = o2::conf::SimConfig::Instance();

  TString dir = getenv("VMCWORKDIR");
  TString geom_dir = dir + "/Detectors/Geometry/";
  gSystem->Setenv("GEOMPATH", geom_dir.Data());

  TString tut_configdir = dir + "/Detectors/gconfig";
  gSystem->Setenv("CONFIG_DIR", tut_configdir.Data());

  // Create simulation run if it does not exist
  if (run == nullptr) {
    run = new FairRunSim();
    run->SetSink(new FairRootFileSink("foo.root")); // Output file
    run->SetName("TGeant3");        // Transport engine
  }
  // Create media
  run->SetMaterials("media.geo"); // Materials

  // we need a field to properly init the media
  run->SetField(o2::base::SimFieldUtils::createMagField());

  // Create geometry
  // we always need the cave
  o2::passive::Cave* cave = new o2::passive::Cave("CAVE");
  // adjust size depending on content
  cave->includeZDC(isActivated("ZDC"));
  // the experiment hall (cave)
  cave->SetGeometryFileName("cave.geo");
  run->AddModule(cave);

  // the experimental hall
  if (isActivated("HALL")) {
    auto hall = new o2::passive::Hall("HALL", "Experimental Hall");
    run->AddModule(hall);
  }

  // the magnet
  if (isActivated("MAG")) {
    // the frame structure to support other detectors
    auto magnet = new o2::passive::Magnet("MAG", "L3 Magnet");
    run->AddModule(magnet);
  }

  // the dipole
  if (isActivated("DIPO")) {
    auto dipole = new o2::passive::Dipole("DIPO", "Alice Dipole");
    run->AddModule(dipole);
  }

  // the compensator dipole
  if (isActivated("COMP")) {
    run->AddModule(new o2::passive::Compensator("COMP", "Alice Compensator Dipole"));
  }

  // beam pipe
  if (isActivated("PIPE")) {
#ifdef ENABLE_UPGRADES
    if (isActivated("IT3")) {
      run->AddModule(new o2::passive::Pipe("PIPE", "Beam pipe", 1.6f, 0.05f));
    } else {
      run->AddModule(new o2::passive::Pipe("PIPE", "Beam pipe"));
    }
#else
    run->AddModule(new o2::passive::Pipe("PIPE", "Beam pipe"));
#endif
  }

#ifdef ENABLE_UPGRADES
  // upgraded beampipe at the interaction point (IP)
  if (isActivated("A3IP")) {
    run->AddModule(new o2::passive::Alice3Pipe("A3IP", "Alice 3 beam pipe", !isActivated("TRK"), 0.48f, 0.015f, 1000.f, 3.7f, 0.05f, 1000.f));
  }
#endif

  // the absorber
  if (isActivated("ABSO")) {
    // the frame structure to support other detectors
    run->AddModule(new o2::passive::Absorber("ABSO", "Absorber"));
  }

  // the shil
  if (isActivated("SHIL")) {
    run->AddModule(new o2::passive::Shil("SHIL", "Small angle beam shield"));
  }

  if (isActivated("TOF") || isActivated("TRD") || isActivated("FRAME")) {
    // the frame structure to support other detectors
    run->AddModule(new o2::passive::FrameStructure("FRAME", "Frame"));
  }

  if (isActivated("TOF")) {
    // TOF
    run->AddModule(new o2::tof::Detector(isReadout("TOF")));
  }

  if (isActivated("TRD")) {
    // TRD
    run->AddModule(new o2::trd::Detector(isReadout("TRD")));
  }

  if (isActivated("TPC")) {
    // tpc
    run->AddModule(new o2::tpc::Detector(isReadout("TPC")));
  }
#ifdef ENABLE_UPGRADES
  if (isActivated("IT3")) {
    // ITS3
    run->AddModule(new o2::its3::Detector(isReadout("IT3")));
  }

  if (isActivated("TRK")) {
    // ALICE 3 TRK
    run->AddModule(new o2::trk::Detector(isReadout("TRK")));
  }

  if (isActivated("FT3")) {
    // ALICE 3 FT3
    run->AddModule(new o2::ft3::Detector(isReadout("FT3")));
  }

  if (isActivated("FCT")) {
    // ALICE 3 FCT
    run->AddModule(new o2::fct::Detector(isReadout("FCT")));
  }
#endif

  if (isActivated("ITS")) {
    // its
    run->AddModule(new o2::its::Detector(isReadout("ITS")));
  }

  if (isActivated("MFT")) {
    // mft
    run->AddModule(new o2::mft::Detector(isReadout("MFT")));
  }

  if (isActivated("MCH")) {
    // mch
    run->AddModule(new o2::mch::Detector(isReadout("MCH")));
  }

  if (isActivated("MID")) {
    // mid
    run->AddModule(new o2::mid::Detector(isReadout("MID")));
  }

  if (isActivated("EMC")) {
    // emcal
    run->AddModule(new o2::emcal::Detector(isReadout("EMC")));
  }

  if (isActivated("PHS")) {
    // phos
    run->AddModule(new o2::phos::Detector(isReadout("PHS")));
  }

  if (isActivated("CPV")) {
    // cpv
    run->AddModule(new o2::cpv::Detector(isReadout("CPV")));
  }

  if (isActivated("FT0")) {
    // FIT-T0
    run->AddModule(new o2::ft0::Detector(isReadout("FT0")));
  }

  if (isActivated("FV0")) {
    // FIT-V0
    run->AddModule(new o2::fv0::Detector(isReadout("FV0")));
  }

  if (isActivated("FDD")) {
    // FIT-FDD
    run->AddModule(new o2::fdd::Detector(isReadout("FDD")));
  }

  if (isActivated("HMP")) {
    // HMP
    run->AddModule(new o2::hmpid::Detector(isReadout("HMP")));
  }

  if (isActivated("ZDC")) {
    // ZDC
    run->AddModule(new o2::zdc::Detector(isReadout("ZDC")));
  }

  if (geomonly) {
    run->Init();
  }
}
