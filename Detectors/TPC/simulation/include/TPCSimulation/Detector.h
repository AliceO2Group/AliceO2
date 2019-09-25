// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef AliceO2_TPC_Detector_H_
#define AliceO2_TPC_Detector_H_

#include "DetectorsBase/Detector.h" // for Detector
#include "Rtypes.h"                 // for Int_t, Double32_t, Double_t, Bool_t, etc
#include "TLorentzVector.h"         // for TLorentzVector
#include "TString.h"

#include "TPCSimulation/Point.h"
#include "TPCBase/Sector.h"

class FairVolume; // lines 10-10

namespace o2
{
namespace tpc
{

class Detector : public o2::base::DetImpl<Detector>
{

 public:
  /** Local material/media IDs for TPC */
  enum EMedium {
    kAir = 0,
    kDriftGas1 = 1,
    kDriftGas2 = 2,
    kCO2 = 3,
    kDriftGas3 = 20,
    kAl = 4,
    kKevlar = 5,
    kNomex = 6,
    kMakrolon = 7,
    kMylar = 8,
    kTedlar = 9,
    kPrepreg1 = 10,
    kPrepreg2 = 11,
    kPrepreg3 = 12,
    kEpoxy = 13,
    kCu = 14,
    kSi = 15,
    kG10 = 16,
    kPlexiglas = 17,
    kSteel = 18,
    kPeek = 19,
    kAlumina = 21,
    kWater = 22,
    kBrass = 23,
    kEpoxyfm = 24,
    kEpoxy1 = 25,
    kAlumina1 = 26
  };
  /**      Name :  Detector Name
    *       Active: kTRUE for active detectors (ProcessHits() will be called)
    *               kFALSE for inactive detectors
    */
  Detector(Bool_t Active);

  /**      default constructor    */
  Detector();

  /**       destructor     */
  ~Detector() override;

  /**      Initialization of the detector is done here    */
  void InitializeO2Detector() override;

  /**       this method is called for each step during simulation
    *       (see FairMCApplication::Stepping())
    */
  //     virtual Bool_t ProcessHitsOrig( FairVolume* v=0);
  Bool_t ProcessHits(FairVolume* v = nullptr) override;

  /**       Registers the produced collections in FAIRRootManager.     */
  void Register() override;

  /** Get the produced hits */
  std::vector<HitGroup>* getHits(Int_t iColl) const
  {
    if (iColl >= 0 && iColl < Sector::MAXSECTOR) {
      return mHitsPerSectorCollection[iColl];
    }
    return nullptr;
  }

  /** tell the branch names corresponding to hits **/
  std::string getHitBranchNames(int coll) const override;

  /**      has to be called after each event to reset the containers      */
  void Reset() override;

  /**      Create the detector geometry        */
  void ConstructGeometry() override;

  /**      This method is an example of how to add your own point
     *       of type DetectorPoint to the clones array
    */
  Point* addHit(float x, float y, float z, float time, float nElectrons, float trackID, float detID);

  /// Copied from AliRoot - should go to someplace else

  /// Empirical ALEPH parameterization of the Bethe-Bloch formula, normalized to 1 at the minimum.
  /// @param bg Beta*Gamma of the incident particle
  /// @param kp* Parameters for the ALICE TPC
  /// @return Bethe-Bloch value in MIP units
  template <typename T>
  T BetheBlochAleph(T bg, T kp1, T kp2, T kp3, T kp4, T kp5);

  /// Copied from AliRoot - should go to someplace else
  /// Function to generate random numbers according to Gamma function
  /// From Hisashi Tanizaki:
  /// http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.158.3866&rep=rep1&type=pdf
  /// Implemented by A. Morsch 14/01/2014
  /// @k is the mean and variance
  Double_t Gamma(Double_t k);

  /** The following methods can be implemented if you need to make
     *  any optional action in your detector during the transport.
    */

  /// Special Geant3? limits and definitions
  /// \todo Check how to deal with this in O2 compared to AliRoot
  /// \todo Discuss in a wider scope
  /// \todo Check correctness of the implementation
  void SetSpecialPhysicsCuts() override;

  void EndOfEvent() override;
  void FinishPrimary() override { ; }
  void FinishRun() override { ; }
  void BeginPrimary() override { ; }
  void PostTrack() override { ; }
  void PreTrack() override { ; }

  void SetGeoFileName(const TString file) { mGeoFileName = file; }
  const TString& GetGeoFileName() const { return mGeoFileName; }

 private:
  int mHitCounter = 0;
  int mElectronCounter = 0;
  int mStepCounter = 0;

  /// Create the detector materials
  virtual void CreateMaterials();

  /// Geant settings hack
  /// \todo Check if still needed see comment in \ref SetSpecialPhysicsCuts
  void GeantHack();

  /// Construct the detector geometry
  void LoadGeometryFromFile();
  /// Construct the detector geometry
  void ConstructTPCGeometry();

  /** Define the sensitive volumes of the geometry */
  void defineSensitiveVolumes();

  // needed by base implementation
  bool setHits(Int_t iColl, std::vector<HitGroup>* ptr)
  {
    if (iColl >= 0 && iColl < Sector::MAXSECTOR) {
      mHitsPerSectorCollection[iColl] = ptr;
      // more entries to set?
      if (iColl < Sector::MAXSECTOR - 1) {
        return true;
      }
    }
    return false;
  }

  /** container for produced hits */
  std::vector<HitGroup>* mHitsPerSectorCollection[Sector::MAXSECTOR]; //! container that keeps track-grouped hits per sector

  TString mGeoFileName; ///< Name of the file containing the TPC geometry
  // size_t mEventNr;                       //!< current event number
  // Events are not successive in MT mode

  /// copy constructor (used in MT)
  Detector(const Detector& rhs);
  Detector& operator=(const Detector&);

  template <typename Det>
  friend class o2::base::DetImpl;
  ClassDefOverride(Detector, 1);
};

template <typename T>
inline T Detector::BetheBlochAleph(T bg, T kp1, T kp2, T kp3, T kp4, T kp5)
{
  T beta = bg / std::sqrt(static_cast<T>(1.) + bg * bg);

  T aa = std::pow(beta, kp4);
  T bb = std::pow(static_cast<T>(1.) / bg, kp5);
  bb = std::log(kp3 + bb);

  return (kp2 - aa - bb) * kp1 / aa;
}

} // namespace tpc
} // namespace o2

#ifdef USESHM
namespace o2
{
namespace base
{
template <>
struct UseShm<o2::tpc::Detector> {
  static constexpr bool value = true;
};
} // namespace base
} // namespace o2
#endif

#endif // AliceO2_TPC_Detector_H_
