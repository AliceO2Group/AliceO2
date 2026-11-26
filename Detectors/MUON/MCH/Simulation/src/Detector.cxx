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

#include "MCHSimulation/Detector.h"
#include "MCHGeometryCreator/Geometry.h"
#include "DetectorsBase/Stack.h"
#include "Stepper.h"
#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TGeoParallelWorld.h"
#include "TGeoPhysicalNode.h"
#include <fairlogger/Logger.h>
#include <sstream>
#include <iostream>
#include <array>
#include <set>
#include <string>
#include <string_view>

ClassImp(o2::mch::Detector);

namespace o2
{
namespace mch
{

Detector::Detector(bool active)
  : o2::base::DetImpl<Detector>("MCH", active), mStepper{new o2::mch::Stepper}
{
}

Detector::Detector(const Detector& rhs)
  : o2::base::DetImpl<Detector>(rhs), mStepper{new o2::mch::Stepper}
{
}

Detector::~Detector()
{
  delete mStepper;
}

void Detector::defineSensitiveVolumes()
{
  for (auto* vol : geo::getSensitiveVolumes()) {
    AddSensitiveVolume(vol);
  }
}

void Detector::InitializeO2Detector()
{
  defineSensitiveVolumes();
}

void Detector::ConstructGeometry()
{
  TGeoVolume* top = gGeoManager->GetTopVolume();
  if (!top) {
    throw std::runtime_error("Cannot create MCH geometry without a top volume");
  }
  geo::createGeometry(*gGeoManager, *top);
}

void Detector::addAlignableVolumes() const
{
  if (!gGeoManager) {
    throw std::runtime_error("Cannot add alignable volumes without TGeoManager");
  }
  geo::addAlignableVolumes(*gGeoManager);
}

namespace
{
/// Whether a matrix mirrors space
bool isReflection(TGeoMatrix const& matrix)
{
  const double* r = matrix.GetRotationMatrix();
  return r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) + r[2] * (r[3] * r[7] - r[4] * r[6]) < 0;
}

/// Registers the gas volumes of the MCH detection elements in the parallel world
struct GasRegistrar {
  TGeoParallelWorld& pw;
  std::set<TGeoVolume const*> gas;
  std::set<TGeoVolume const*> registered;
  std::set<std::string> mirroredDEs;
  int nAdded = 0;

  /// Walks the assemblies below path (a detection element is one) and adds the gas volumes in them
  void add(TGeoVolume const& volume, std::string const& path, bool reflected, std::string_view de)
  {
    for (int i = 0; i < volume.GetNdaughters(); ++i) {
      auto node = volume.GetNode(i);
      auto daughter = node->GetVolume();
      auto daughterPath = path + "/" + node->GetName();
      bool daughterReflected = reflected != isReflection(*node->GetMatrix());
      if (daughter->IsAssembly()) {
        add(*daughter, daughterPath, daughterReflected, de);
      } else if (gas.contains(daughter)) {
        // a mirrored node in the parallel world stops tracks at its boundary with TGeo in Geant4,
        // and VecGeomParallelWorld finds no VecGeom path for it (fatal)
        if (daughterReflected) {
          mirroredDEs.emplace(de);
          continue;
        }
        pw.AddNode(daughterPath.c_str());
        registered.insert(daughter);
        ++nAdded;
      }
    }
  }
};
} // namespace

void Detector::fillParallelWorld() const
{
  TGeoParallelWorld* pw = gGeoManager->GetParallelWorld();
  if (pw == nullptr) {
    LOG(error) << "Parallel world was not created";
    return;
  }
  auto sensitive = geo::getSensitiveVolumes();
  GasRegistrar registrar{*pw, std::set<TGeoVolume const*>(sensitive.begin(), sensitive.end())};
  for (int i = 0; i < gGeoManager->GetNAlignable(); ++i) {
    auto pne = gGeoManager->GetAlignableEntry(i);
    std::string_view sname = pne->GetName();
    if (!sname.starts_with("MCH/HC") || sname.find("/DE") == std::string_view::npos) {
      continue;
    }
    if (!gGeoManager->cd(pne->GetTitle())) {
      LOG(fatal) << "Cannot reach " << sname << " at " << pne->GetTitle();
    }
    registrar.add(*gGeoManager->GetCurrentVolume(), pne->GetTitle(), isReflection(*gGeoManager->GetCurrentMatrix()),
                  sname.substr(sname.rfind('/') + 1));
  }
  gGeoManager->CdTop();
  LOG(info) << "MCH added " << registrar.nAdded << " gas volumes to the parallel world";
  for (auto const& de : registrar.mirroredDEs) {
    LOG(warning) << "MCH left the gas of the mirrored placement " << de << " out of the parallel world";
  }
  for (auto* vol : sensitive) {
    if (!registrar.registered.contains(vol)) {
      LOG(error) << "MCH sensitive volume " << vol->GetName() << " was not added to the parallel world";
    }
  }
}

Bool_t Detector::ProcessHits(FairVolume* v)
{
  mStepper->process(*fMC);
  (static_cast<o2::data::Stack*>(fMC->GetStack()))->addHit(GetDetId());
  return kTRUE;
}

std::vector<o2::mch::Hit>* Detector::getHits(int i)
{
  if (i == 0) {
    return mStepper->getHits();
  }
  return nullptr;
}

bool Detector::setHits(int i, std::vector<o2::mch::Hit>* ptr)
{
  if (i == 0) {
    mStepper->setHits(ptr);
  }
  return false;
}

void Detector::Register()
{
  // TODO : get another way to do I/O (i.e. separate concerns)

  mStepper->registerHits(addNameTo("Hit").c_str());
}

void Detector::EndOfEvent() { mStepper->resetHits(); }

} // namespace mch
} // namespace o2
