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

#ifndef ALICEO2_EC0_DETECTOR_H_
#define ALICEO2_EC0_DETECTOR_H_

#include <vector>                             // for vector
#include "DetectorsBase/GeometryManager.h"    // for getSensID
#include "DetectorsBase/Detector.h"           // for Detector
#include "DetectorsCommonDataFormats/DetID.h" // for Detector
#include "ITSMFTSimulation/Hit.h"             // for Hit
#include "Rtypes.h"                           // for Int_t, Double_t, Float_t, Bool_t, etc
#include "TArrayD.h"                          // for TArrayD
#include "TGeoManager.h"                      // for gGeoManager, TGeoManager (ptr only)
#include "TLorentzVector.h"                   // for TLorentzVector
#include "TVector3.h"                         // for TVector3

class FairVolume;
class TGeoVolume;

class TParticle;

class TString;

namespace o2
{
namespace ec0
{
class GeometryTGeo;
}
} // namespace o2
namespace o2
{
namespace ec0
{
class EC0Layer;
}
} // namespace o2

namespace o2
{
namespace ec0
{
class EC0Layer;

class Detector : public o2::base::DetImpl<Detector>
{
 public:
  /// Name : Detector Name
  /// Active: kTRUE for active detectors (ProcessHits() will be called)
  ///         kFALSE for inactive detectors
  Detector(Bool_t active);

  /// Default constructor
  Detector();

  /// Default destructor
  ~Detector() override;

  /// Initialization of the detector is done here
  void InitializeO2Detector() override;

  /// This method is called for each step during simulation (see FairMCApplication::Stepping())
  Bool_t ProcessHits(FairVolume* v = nullptr) override;

  /// Registers the produced collections in FAIRRootManager
  void Register() override;

  /// Gets the produced collections
  std::vector<o2::itsmft::Hit>* getHits(Int_t iColl) const
  {
    if (iColl == 0) {
      return mHits;
    }
    return nullptr;
  }

 public:
  /// Has to be called after each event to reset the containers
  void Reset() override;

  /// Base class to create the detector geometry
  void ConstructGeometry() override;

  /// This method is an example of how to add your own point of type Hit to the clones array
  o2::itsmft::Hit* addHit(int trackID, int detID, const TVector3& startPos, const TVector3& endPos,
                          const TVector3& startMom, double startE, double endTime, double eLoss,
                          unsigned char startStatus, unsigned char endStatus);

  /// Add alignable top volumes
  void addAlignableVolumes() const override;
  void addAlignableVolumesLayer(Int_t lr, TString& parent, Int_t& lastUID) const;
  void addAlignableVolumesChip(Int_t lr, TString& parent, Int_t& lastUID) const;
  void addAlignableVolumesSensor(Int_t lr, TString& parent, Int_t& lastUID) const;

  Int_t chipVolUID(Int_t id) const { return o2::base::GeometryManager::getSensID(o2::detectors::DetID::EC0, id); }

  void EndOfEvent() override;

  void FinishPrimary() override { ; }
  virtual void finishRun() { ; }
  void BeginPrimary() override { ; }
  void PostTrack() override { ; }
  void PreTrack() override { ; }
  /// Prints out the content of this class in ASCII format
  /// \param ostream *os The output stream
  void Print(std::ostream* os) const;

  /// Reads in the content of this class in the format of Print
  /// \param istream *is The input stream
  void Read(std::istream* is);

  /// Returns the number of layers
  Int_t getNumberOfLayers() const { return mNumberOfLayers; }

  void buildBasicEC0(int nLayers = 10, Float_t z_first = -16.0, Float_t z_length = 263, Float_t etaIn = -4.5, Float_t etaOut = -1.5, Float_t Layerx2X0 = 0.01);
  void buildEC0V1();

  GeometryTGeo* mGeometryTGeo; //! access to geometry details

 protected:
  std::vector<Int_t> mLayerID;
  std::vector<TString> mLayerName;
  Int_t mNumberOfLayers;

 private:
  /// this is transient data about track passing the sensor
  struct TrackData {               // this is transient
    bool mHitStarted;              //! hit creation started
    unsigned char mTrkStatusStart; //! track status flag
    TLorentzVector mPositionStart; //! position at entrance
    TLorentzVector mMomentumStart; //! momentum
    double mEnergyLoss;            //! energy loss
  } mTrackData;                    //!

  /// Container for hit data
  std::vector<o2::itsmft::Hit>* mHits;

  /// Create the detector materials
  virtual void createMaterials();

  /// Create the detector geometry
  void createGeometry();

  /// Define the sensitive volumes of the geometry
  void defineSensitiveVolumes();

  Detector(const Detector&);

  Detector& operator=(const Detector&);

  std::vector<EC0Layer> mLayers;

  template <typename Det>
  friend class o2::base::DetImpl;
  ClassDefOverride(Detector, 1);
};

// Input and output function for standard C++ input/output.
std::ostream& operator<<(std::ostream& os, Detector& source);

std::istream& operator>>(std::istream& os, Detector& source);
} // namespace ec0
} // namespace o2

#ifdef USESHM
namespace o2
{
namespace base
{
template <>
struct UseShm<o2::ec0::Detector> {
  static constexpr bool value = true;
};
} // namespace base
} // namespace o2
#endif

#endif
