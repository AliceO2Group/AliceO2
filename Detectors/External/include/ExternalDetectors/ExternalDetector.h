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

/// \file ExternalDetector.h
/// \brief Sensitive detector built from an externally provided (CAD-derived) geometry
///
/// ExternalDetector is the sensitive counterpart of o2::passive::ExternalModule.
/// It injects a CAD-derived TGeo geometry (produced by scripts/geometry/O2_CADtoTGeo.py)
/// and turns a configurable set of its volumes (selected by their medium name) into
/// sensitive volumes which produce hits. It derives from o2::base::DetImpl, so it
/// transparently participates in the full o2-sim hit forwarding/merging machinery
/// (FairMQ serialization, sub-event merging, ...).
///
/// As discussed, the detector identity (hit output format) is tied to an existing
/// o2::detectors::DetID. For this first demonstrator it borrows the ITS hit type
/// (o2::itsmft::Hit) and routes its hits to a configurable DetID (ITS by default).

#ifndef ALICEO2_EXT_EXTERNALDETECTOR_H
#define ALICEO2_EXT_EXTERNALDETECTOR_H

#include "DetectorsBase/Detector.h"            // for DetImpl
#include "DetectorsCommonDataFormats/DetID.h"  // for DetID
#include "ITSMFTSimulation/Hit.h"              // for the (borrowed) hit type

#include "Rtypes.h"
#include "TLorentzVector.h"

#include <set>
#include <string>
#include <unordered_map>
#include <vector>

class FairVolume;
class TGeoMatrix;
class TVector3;

namespace o2::ext
{

/// Configuration of a single sensitive external detector.
struct ExternalDetectorOptions {
  std::string root_macro_file;               // ROOT macro describing the CAD geometry (O2_CADtoTGeo.py output)
  std::string anchor_volume;                 // existing volume into which the geometry is hooked
  TGeoMatrix const* placement = nullptr;     // placement of the geometry inside the anchor (may be null)
  std::vector<std::string> sensitiveMedia;   // media (substring match on the medium name) to be made sensitive
  std::vector<std::string> sensitiveVolumes; // volumes (substring match on the volume name) to be made sensitive
  int detID = o2::detectors::DetID::ITS;     // DetID this detector's hits are tied to (identity / output format)
};

class ExternalDetector : public o2::base::DetImpl<ExternalDetector>
{
 public:
  ExternalDetector(const char* name, const char* title, ExternalDetectorOptions options);
  ExternalDetector();
  ~ExternalDetector() override;

  /// Build a list of sensitive external detectors from a JSON description file.
  /// The file must contain an "externalDetectors" array; each entry needs at least
  /// "name", "macro", "anchor" and at least one of "sensitiveMedia" / "sensitiveVolumes"
  /// (arrays of substrings matched against medium / volume names); an optional
  /// "detID" (name, default "ITS") ties the hit output to an existing detector, and
  /// an optional "placement" object may carry "translation"/"rotation_deg".
  /// Ownership of the returned detectors is transferred to the caller.
  static std::vector<ExternalDetector*> createFromJSON(const std::string& jsonfile);

  /// Build the CAD geometry, remap its media and register the sensitive volumes.
  void ConstructGeometry() override;

  /// Resolve the Monte Carlo volume IDs of the sensitive volumes.
  void InitializeO2Detector() override;

  /// Called for each tracking step; produces hits in the sensitive volumes.
  Bool_t ProcessHits(FairVolume* v = nullptr) override;

  /// Register the hit collection with the FairRootManager.
  void Register() override;

  /// Get the produced hit collection (probe interface used by DetImpl).
  std::vector<o2::itsmft::Hit>* getHits(Int_t iColl) const
  {
    if (iColl == 0) {
      return mHits;
    }
    return nullptr;
  }

  void Reset() override;
  void EndOfEvent() override;

  void FinishPrimary() override {}
  void BeginPrimary() override {}
  void PostTrack() override {}
  void PreTrack() override {}

 protected:
  o2::itsmft::Hit* addHit(int trackID, int detID, const TVector3& startPos, const TVector3& endPos,
                          const TVector3& startMom, double startE, double endTime, double eLoss,
                          unsigned char startStatus, unsigned char endStatus);

  /// recursively collect names of volumes whose medium matches the configured sensitive media
  void collectSensitiveVolumeNames(TGeoVolume* vol, std::set<TGeoVolume*>& visited);

  ExternalDetectorOptions mOptions;

  std::vector<std::string> mSensitiveVolumeNames; //! names of the volumes to be made sensitive (filled at geometry build)
  std::set<int> mSensitiveVolIDs;                 //! MC volume IDs of the sensitive volumes
  std::unordered_map<int, int> mVolID2SensorID;   //! dense sensor index per sensitive MC volume ID

  /// transient data about a track passing a sensor (mirrors the ITS approach)
  struct TrackData {
    bool mHitStarted;              //! hit creation started
    unsigned char mTrkStatusStart; //! track status flag at entrance
    TLorentzVector mPositionStart; //! position at entrance
    TLorentzVector mMomentumStart; //! momentum at entrance
    double mEnergyLoss;            //! accumulated energy loss
  } mTrackData;                    //!

  std::vector<o2::itsmft::Hit>* mHits = nullptr; //! container for produced hits

  int mStepCount = 0; //! number of stepping calls inside our sensitive volumes this event (probe)

 private:
  ExternalDetector(const ExternalDetector&);
  ExternalDetector& operator=(const ExternalDetector&);

  template <typename Det>
  friend class o2::base::DetImpl;
  ClassDefOverride(ExternalDetector, 1);
};

} // namespace o2::ext

#endif
