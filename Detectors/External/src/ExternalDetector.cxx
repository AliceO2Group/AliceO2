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

#include "ExternalDetectors/ExternalDetector.h"
#include "DetectorsBase/CADGeometryUtils.h"
#include "DetectorsBase/Stack.h"
#include "CommonUtils/FileSystemUtils.h"
#include "CommonUtils/ShmManager.h"
#include "CommonUtils/ShmAllocator.h"

#include <FairRootManager.h>
#include <FairVolume.h>
#include <fairlogger/Logger.h>

#include <TGeoManager.h>
#include <TGeoMatrix.h>
#include <TGeoMedium.h>
#include <TGeoNode.h>
#include <TGeoVolume.h>
#include <TVirtualMC.h>
#include <TVector3.h>

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>

#include <fstream>

namespace o2::ext
{

ExternalDetector::ExternalDetector(const char* name, const char* title, ExternalDetectorOptions options)
  : o2::base::DetImpl<ExternalDetector>(name, true),
    mOptions(options),
    mTrackData(),
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>())
{
  (void)title; // the FairModule title is the second base ctor argument; kept for symmetry with other detectors
  // Decouple the user-facing FairModule name (e.g. "IRIS") from the DetId: the base
  // ctor derives fDetId from the name which is generally not a registered DetID, so we
  // explicitly tie this detector to the configured (existing) DetID. This is what makes
  // the hit output format / identity well defined, as discussed.
  fDetId = mOptions.detID;
}

ExternalDetector::ExternalDetector()
  : o2::base::DetImpl<ExternalDetector>("EXTDET", true),
    mTrackData(),
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>())
{
}

ExternalDetector::ExternalDetector(const ExternalDetector& rhs)
  : o2::base::DetImpl<ExternalDetector>(rhs),
    mOptions(rhs.mOptions),
    mSensitiveVolumeNames(rhs.mSensitiveVolumeNames),
    mSensitiveVolIDs(rhs.mSensitiveVolIDs),
    mVolID2SensorID(rhs.mVolID2SensorID),
    mTrackData(),
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>())
{
}

ExternalDetector::~ExternalDetector()
{
  if (mHits) {
    o2::utils::freeSimVector(mHits);
  }
}

void ExternalDetector::collectSensitiveVolumeNames(TGeoVolume* vol, std::set<TGeoVolume*>& visited)
{
  if (!vol || visited.count(vol)) {
    return;
  }
  visited.insert(vol);

  bool sensitive = false;
  // match by volume name
  const std::string volname = vol->GetName();
  for (const auto& token : mOptions.sensitiveVolumes) {
    if (!token.empty() && volname.find(token) != std::string::npos) {
      sensitive = true;
      break;
    }
  }
  // otherwise match by medium name
  if (!sensitive) {
    if (auto medium = vol->GetMedium()) {
      const std::string medname = medium->GetName();
      for (const auto& token : mOptions.sensitiveMedia) {
        if (!token.empty() && medname.find(token) != std::string::npos) {
          sensitive = true;
          break;
        }
      }
    }
  }
  if (sensitive) {
    mSensitiveVolumeNames.emplace_back(volname);
  }

  const int nd = vol->GetNdaughters();
  for (int i = 0; i < nd; ++i) {
    if (auto node = vol->GetNode(i)) {
      collectSensitiveVolumeNames(node->GetVolume(), visited);
    }
  }
}

void ExternalDetector::ConstructGeometry()
{
  // build the CAD geometry and obtain its top volume
  auto module_top = o2::base::buildCADVolumeFromMacro(mOptions.root_macro_file, GetName());
  if (!module_top) {
    LOG(error) << "No geometry could be built for external detector " << GetName();
    return;
  }

  // bring the CAD media under O2's MaterialManager
  o2::base::remapCADMedia(module_top, GetName());

  // determine which volumes should become sensitive (selected by medium name)
  mSensitiveVolumeNames.clear();
  std::set<TGeoVolume*> visited;
  collectSensitiveVolumeNames(module_top, visited);
  if (mSensitiveVolumeNames.empty()) {
    LOG(warning) << "External detector " << GetName() << ": no volume matched the configured sensitive media; "
                 << "no hits will be produced";
  } else {
    LOG(info) << "External detector " << GetName() << ": " << mSensitiveVolumeNames.size()
              << " sensitive volume(s) selected";
  }

  // place it into the provided anchor volume (needs to exist)
  auto anchor = gGeoManager->FindVolumeFast(mOptions.anchor_volume.c_str());
  if (!anchor) {
    LOG(error) << "Anchor volume " << mOptions.anchor_volume << " not found. Aborting";
    return;
  }
  anchor->AddNode(module_top, 1, const_cast<TGeoMatrix*>(mOptions.placement));
}

void ExternalDetector::InitializeO2Detector()
{
  // resolve the MC volume IDs of the sensitive volumes and register them with FairRoot
  mSensitiveVolIDs.clear();
  mVolID2SensorID.clear();
  int sensorID = 0;
  for (const auto& name : mSensitiveVolumeNames) {
    const int volID = registerSensitiveVolumeAndGetVolID(name);
    if (volID <= 0) {
      continue;
    }
    mSensitiveVolIDs.insert(volID);
    mVolID2SensorID[volID] = sensorID++;
    LOG(info) << "External detector " << GetName() << ": registered sensitive volume '" << name
              << "' (MC volID " << volID << ", sensor " << mVolID2SensorID[volID] << ")";
  }
}

Bool_t ExternalDetector::ProcessHits(FairVolume* vol)
{
  // This method is called from the MC stepping for the registered sensitive volumes
  if (!(fMC->TrackCharge())) {
    return kFALSE;
  }

  const int volID = vol ? vol->getMCid() : -1;
  auto sensorIter = mVolID2SensorID.find(volID);
  if (sensorIter == mVolID2SensorID.end()) {
    return kFALSE; // not one of our sensitive volumes
  }
  ++mStepCount; // probe: count stepping calls inside our sensitive volumes

  bool startHit = false, stopHit = false;
  unsigned char status = 0;
  if (fMC->IsTrackEntering()) {
    status |= o2::itsmft::Hit::kTrackEntering;
  }
  if (fMC->IsTrackInside()) {
    status |= o2::itsmft::Hit::kTrackInside;
  }
  if (fMC->IsTrackExiting()) {
    status |= o2::itsmft::Hit::kTrackExiting;
  }
  if (fMC->IsTrackOut()) {
    status |= o2::itsmft::Hit::kTrackOut;
  }
  if (fMC->IsTrackStop()) {
    status |= o2::itsmft::Hit::kTrackStopped;
  }
  if (fMC->IsTrackAlive()) {
    status |= o2::itsmft::Hit::kTrackAlive;
  }

  // track is entering or created in the volume
  if ((status & o2::itsmft::Hit::kTrackEntering) || (status & o2::itsmft::Hit::kTrackInside && !mTrackData.mHitStarted)) {
    startHit = true;
  } else if ((status & (o2::itsmft::Hit::kTrackExiting | o2::itsmft::Hit::kTrackOut | o2::itsmft::Hit::kTrackStopped))) {
    stopHit = true;
  }

  // increment energy loss at all steps except entrance
  if (!startHit) {
    mTrackData.mEnergyLoss += fMC->Edep();
  }
  if (!(startHit | stopHit)) {
    return kFALSE; // do nothing
  }

  if (startHit) {
    mTrackData.mEnergyLoss = 0.;
    fMC->TrackMomentum(mTrackData.mMomentumStart);
    fMC->TrackPosition(mTrackData.mPositionStart);
    mTrackData.mTrkStatusStart = status;
    mTrackData.mHitStarted = true;
  }
  if (stopHit) {
    TLorentzVector positionStop;
    fMC->TrackPosition(positionStop);

    auto stack = static_cast<o2::data::Stack*>(fMC->GetStack());
    addHit(stack->GetCurrentTrackNumber(), sensorIter->second, mTrackData.mPositionStart.Vect(), positionStop.Vect(),
           mTrackData.mMomentumStart.Vect(), mTrackData.mMomentumStart.E(), positionStop.T(),
           mTrackData.mEnergyLoss, mTrackData.mTrkStatusStart, status);
    mTrackData.mHitStarted = false;

    // register that this track left a hit in our detector (sets the hit bit on the MCTrack)
    stack->addHit(GetDetId());
  }
  return kTRUE;
}

o2::itsmft::Hit* ExternalDetector::addHit(int trackID, int detID, const TVector3& startPos, const TVector3& endPos,
                                          const TVector3& startMom, double startE, double endTime, double eLoss,
                                          unsigned char startStatus, unsigned char endStatus)
{
  mHits->emplace_back(trackID, detID, startPos, endPos, startMom, startE, endTime, eLoss, startStatus, endStatus);
  return &(mHits->back());
}

void ExternalDetector::Register()
{
  // Create a branch (named "<name>Hit") holding the produced hits.
  if (FairRootManager::Instance()) {
    FairRootManager::Instance()->RegisterAny(addNameTo("Hit").data(), mHits, kTRUE);
  }
}

void ExternalDetector::Reset()
{
  if (!o2::utils::ShmManager::Instance().isOperational()) {
    mHits->clear();
  }
}

void ExternalDetector::EndOfEvent()
{
  // probe: report how often our sensitive volumes were stepped through and how many hits resulted
  LOG(info) << "External detector " << GetName() << " EndOfEvent: " << mStepCount
            << " sensitive step(s) -> " << (mHits ? mHits->size() : 0) << " hit(s)";
  mStepCount = 0;
  Reset();
}

namespace
{
// Build a TGeoCombiTrans from an optional JSON "placement" object carrying
// "translation":[x,y,z] (cm) and/or "rotation_deg":[rx,ry,rz] (deg, applied X,Y,Z).
TGeoMatrix* makePlacementFromJSON(const rapidjson::Value& placement)
{
  auto combi = new TGeoCombiTrans();
  if (placement.HasMember("rotation_deg") && placement["rotation_deg"].IsArray()) {
    const auto& r = placement["rotation_deg"];
    if (r.Size() == 3) {
      combi->RotateX(r[0].GetDouble());
      combi->RotateY(r[1].GetDouble());
      combi->RotateZ(r[2].GetDouble());
    } else {
      LOG(warning) << "ExternalDetector placement 'rotation_deg' must have 3 entries; ignoring";
    }
  }
  if (placement.HasMember("translation") && placement["translation"].IsArray()) {
    const auto& t = placement["translation"];
    if (t.Size() == 3) {
      combi->SetDx(t[0].GetDouble());
      combi->SetDy(t[1].GetDouble());
      combi->SetDz(t[2].GetDouble());
    } else {
      LOG(warning) << "ExternalDetector placement 'translation' must have 3 entries; ignoring";
    }
  }
  return combi;
}
} // namespace

std::vector<ExternalDetector*> ExternalDetector::createFromJSON(const std::string& jsonfile)
{
  std::vector<ExternalDetector*> result;

  auto expanded = o2::utils::expandShellVarsInFileName(jsonfile);
  std::ifstream fileStream(expanded, std::ios::in);
  if (!fileStream.is_open()) {
    LOG(error) << "Cannot open external geometry config file '" << expanded << "'";
    return result;
  }

  rapidjson::IStreamWrapper isw(fileStream);
  rapidjson::Document doc;
  doc.ParseStream(isw);
  if (doc.HasParseError()) {
    LOG(error) << "Error parsing external geometry JSON '" << expanded << "': "
               << rapidjson::GetParseError_En(doc.GetParseError())
               << " (offset " << doc.GetErrorOffset() << ")";
    return result;
  }
  // the array of sensitive external detectors is optional (the same file may only
  // configure passive external modules)
  if (!doc.HasMember("externalDetectors")) {
    return result;
  }
  if (!doc["externalDetectors"].IsArray()) {
    LOG(error) << "External geometry JSON '" << expanded << "': 'externalDetectors' must be an array";
    return result;
  }

  auto getString = [](const rapidjson::Value& v, const char* key) -> std::string {
    if (v.HasMember(key) && v[key].IsString()) {
      return v[key].GetString();
    }
    return std::string();
  };

  for (const auto& entry : doc["externalDetectors"].GetArray()) {
    if (!entry.IsObject()) {
      LOG(error) << "Skipping non-object entry in 'externalDetectors'";
      continue;
    }
    const auto name = getString(entry, "name");
    if (name.empty()) {
      LOG(error) << "Skipping external detector entry without 'name'";
      continue;
    }
    ExternalDetectorOptions options;
    options.root_macro_file = getString(entry, "macro");
    options.anchor_volume = getString(entry, "anchor");
    if (options.root_macro_file.empty() || options.anchor_volume.empty()) {
      LOG(error) << "External detector '" << name << "' requires both 'macro' and 'anchor'; skipping";
      continue;
    }

    if (entry.HasMember("sensitiveMedia") && entry["sensitiveMedia"].IsArray()) {
      for (const auto& m : entry["sensitiveMedia"].GetArray()) {
        if (m.IsString()) {
          options.sensitiveMedia.emplace_back(m.GetString());
        }
      }
    }
    if (entry.HasMember("sensitiveVolumes") && entry["sensitiveVolumes"].IsArray()) {
      for (const auto& v : entry["sensitiveVolumes"].GetArray()) {
        if (v.IsString()) {
          options.sensitiveVolumes.emplace_back(v.GetString());
        }
      }
    }
    if (options.sensitiveMedia.empty() && options.sensitiveVolumes.empty()) {
      LOG(error) << "External detector '" << name
                 << "' requires a non-empty 'sensitiveMedia' or 'sensitiveVolumes' array; skipping";
      continue;
    }

    const auto detIDName = getString(entry, "detID");
    if (!detIDName.empty()) {
      const auto did = o2::detectors::DetID::nameToID(detIDName.c_str());
      if (did < 0 || did >= o2::detectors::DetID::nDetectors) {
        LOG(error) << "External detector '" << name << "': unknown detID '" << detIDName << "'; skipping";
        continue;
      }
      options.detID = did;
    }

    if (entry.HasMember("placement") && entry["placement"].IsObject()) {
      options.placement = makePlacementFromJSON(entry["placement"]);
    }

    auto title = getString(entry, "title");
    if (title.empty()) {
      title = name;
    }
    LOG(info) << "Configured external detector '" << name << "' from macro '" << options.root_macro_file
              << "' anchored to '" << options.anchor_volume << "', tied to DetID '"
              << o2::detectors::DetID::getName(options.detID) << "'";
    result.push_back(new ExternalDetector(name.c_str(), title.c_str(), options));
  }
  return result;
}

} // namespace o2::ext

ClassImp(o2::ext::ExternalDetector);
