// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

  //

  Int_t isVersion() const { return mVersion; }
  /// Creating materials for the detector

  void createMaterials();

  enum EMedia {
    Zero,
    Air,
    Vacuum,
    Si,
    Readout,
    Support,
    Carbon,
    Be,
    Alu,
    Water,
    SiO2,
    Inox,
    Kapton,
    Epoxy,
    CarbonFiber,
    CarbonEpoxy,
    Rohacell,
    Polyimide,
    PEEK,
    FR4,
    Cu,
    X7R,
    X7Rw,
    CarbonFleece,
    SE4445
  }; // media IDs used in CreateMaterials

  void setDensitySupportOverSi(Double_t density)
  {
    if (density > 1e-6)
      mDensitySupportOverSi = density;
    else
      mDensitySupportOverSi = 1e-6;
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
