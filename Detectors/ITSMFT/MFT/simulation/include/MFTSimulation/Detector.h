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

/// \file Detector.h
/// \brief Definition of the Detector class
/// \author antonio.uras@cern.ch, bogdan.vulpescu@cern.ch
/// \date 01/08/2016

#ifndef ALICEO2_MFT_DETECTOR_H
#define ALICEO2_MFT_DETECTOR_H

#include "TLorentzVector.h"

#include <vector> // for vector
#include "DetectorsBase/Detector.h"
#include "DetectorsCommonDataFormats/DetID.h" // for Detector
#include "ITSMFTSimulation/Hit.h"             // for Hit

class TVector3;

namespace o2
{
namespace itsmft
{
class Hit;
}
} // namespace o2

namespace o2
{
namespace mft
{
class GeometryTGeo;
}
} // namespace o2

namespace o2
{
namespace mft
{
class Detector : public o2::base::DetImpl<Detector>
{
 public:
  /// Default constructor
  Detector();

  /// Default destructor
  ~Detector() override;

  /// Initialization of the detector is done here
  void InitializeO2Detector() override;

  /// This method is called for each step during simulation (see FairMCApplication::Stepping())
  Bool_t ProcessHits(FairVolume* v = nullptr) override;

  /// Has to be called after each event to reset the containers
  void Reset() override;

  /// Registers the produced collections in FAIRRootManager
  void Register() override;

  /// Gets the produced hits
  std::vector<o2::itsmft::Hit>* getHits(Int_t iColl) const
  {
    if (iColl == 0) {
      return mHits;
    }
    return nullptr;
  }

  void EndOfEvent() override;

  void FinishPrimary() override { ; }
  void FinishRun() override { ; }
  void BeginPrimary() override { ; }
  void PostTrack() override { ; }
  void PreTrack() override { ; }
  void ConstructGeometry() override; // inherited from FairModule

  /// Add alignable top volumes
  void addAlignableVolumes() const override;

  /// Add alignable Half volumes
  /// \param hf Half number
  /// \param parent path of the parent volume
  /// \param lastUID on output, UID of the last volume
  void addAlignableVolumesHalf(Int_t hf, TString& parent, Int_t& lastUID) const;

  /// Add alignable Disk volumes
  /// \param hf half number
  /// \param dk disk number
  /// \param parent path of the parent volume
  /// \param lastUID on output, UID of the last volume
  void addAlignableVolumesDisk(Int_t hf, Int_t dk, TString& parent, Int_t& lastUID) const;

  /// Add alignable Ladder volumes
  /// \param hf half number
  /// \param dk disk number
  /// \param lr ladder stave number
  /// \param parent path of the parent volume
  /// \param lastUID on output, UID of the last volume
  void addAlignableVolumesLadder(Int_t hf, Int_t dk, Int_t lr, TString& parent, Int_t& lastUID) const;

  /// Add alignable Sensor volumes
  /// \param hf half number
  /// \param dk disk number
  /// \param lr ladder number
  /// \param ms sensor number
  /// \param parent path of the parent volume
  /// \param lastUID on output, UID of the last volume
  void addAlignableVolumesChip(Int_t hf, Int_t dk, Int_t lr, Int_t ms, TString& parent, Int_t& lastUID) const;

  Int_t isVersion() const { return mVersion; }
  /// Creating materials for the detector

  void createMaterials();
  void setDensitySupportOverSi(Double_t density)
  {
    if (density > 1e-6) {
      mDensitySupportOverSi = density;
    } else {
      mDensitySupportOverSi = 1e-6;
    }
  }

  void createGeometry();
  void defineSensitiveVolumes();

  GeometryTGeo* mGeometryTGeo; //! access to geometry details

 protected:
  Int_t mVersion;                 //
  Double_t mDensitySupportOverSi; //

 private:
  /// Container for hit data
  std::vector<o2::itsmft::Hit>* mHits;

  Detector(const Detector&);
  Detector& operator=(const Detector&);

  o2::itsmft::Hit* addHit(int trackID, int detID, TVector3 startPos, TVector3 endPos, TVector3 startMom, double startE,
                          double endTime, double eLoss, unsigned char startStatus, unsigned char endStatus);

  /// this is transient data about track passing the sensor
  struct TrackData {               // this is transient
    bool mHitStarted;              //! hit creation started
    unsigned char mTrkStatusStart; //! track status flag
    TLorentzVector mPositionStart; //! position at entrance
    TLorentzVector mMomentumStart; //! momentum
    double mEnergyLoss;            //! energy loss
  } mTrackData;                    //!

  template <typename Det>
  friend class o2::base::DetImpl;
  ClassDefOverride(Detector, 1);
};

// Input and output function for standard C++ input/output.
std::ostream& operator<<(std::ostream& os, Detector& source);
std::istream& operator>>(std::istream& os, Detector& source);
} // namespace mft
} // namespace o2

#ifdef USESHM
namespace o2
{
namespace base
{
template <>
struct UseShm<o2::mft::Detector> {
  static constexpr bool value = true;
};
} // namespace base
} // namespace o2
#endif

#endif
