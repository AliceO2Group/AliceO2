// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_EMCAL_DETECTOR_H_
#define ALICEO2_EMCAL_DETECTOR_H_

#include "MathUtils/Cartesian3D.h"
#include "DetectorsBase/Detector.h"
#include "Rtypes.h"
#include "TArrayF.h"
#include "TString.h"

#include <vector>

class FairVolume;
class TClonesArray;

namespace o2
{
namespace EMCAL
{

class Hit;
class Geometry;
    
class Detector : public o2::Base::Detector
{
 public:
  enum { ID_AIR = 0, ID_PB = 1, ID_SC = 2, ID_AL = 3, ID_STEEL = 4, ID_PAPER = 5 };

  Detector() = default;

  Detector(const char* Name, Bool_t Active);

  ~Detector() override = default;

  void Initialize() final;

  Bool_t ProcessHits(FairVolume* v = nullptr) final;

  Hit* AddHit(Int_t shunt, Int_t trackID, Int_t parentID, Int_t primary, Double_t initialEnergy, Int_t detID,
              const Point3D<float>& pos, const Vector3D<float>& mom, Double_t time, Double_t length);

  void Register() override;

  TClonesArray* GetCollection(Int_t iColl) const final;

  void Reset() final;

  Geometry* GetGeometry();

 protected:
  ///
  /// Creating detector materials for the EMCAL detector and space frame
  ///
  void CreateMaterials();

  void ConstructGeometry() override;

  ///
  /// Generate tower geometry
  ///
  void CreateShiskebabGeometry();

  ///
  /// Generate super module geometry
  ///
  void CreateSmod(const char* mother = "XEN1");

  ///
  /// Generate module geometry (2x2 towers)
  ///
  void CreateEmod(const char* mother = "SMOD", const char* child = "EMOD");

  ///
  /// Generate aluminium plates geometry
  ///
  void CreateAlFrontPlate(const char* mother = "EMOD", const char* child = "ALFP");

  ///
  /// Generate towers in module of 1x1
  /// Prototype studies, remove?
  ///
  void Trd1Tower1X1(Double_t* parSCM0);

  ///
  /// Generate towers in module of 3x3
  /// Prototype studies, remove?
  ///
  void Trd1Tower3X3(const Double_t* parSCM0);

  ///
  /// Used by AliEMCALv0::Trd1Tower3X3
  /// Prototype studies, remove?
  ///
  void PbInTrap(const Double_t parTRAP[11], TString n);

  ///
  /// Used by AliEMCALv0::Trd1Tower1X1
  /// Prototype studies, remove?
  ///
  void PbInTrd1(const Double_t* parTrd1, TString n);

 private:
  Int_t mBirkC0;
  Double_t mBirkC1;
  Double_t mBirkC2;

  TClonesArray* mPointCollection; ///< Collection of EMCAL points
  Geometry* mGeometry;            ///< Geometry pointer

  TArrayF mEnvelop1;         //!<! parameters of EMCAL envelop for TRD1(2) case
  Int_t mIdRotm;             //!<! number of rotation matrix (working variable)

  Double_t mSampleWidth; //!<! sample width = double(g->GetECPbRadThick()+g->GetECScintThick());
  Double_t mSmodPar0;    //!<! x size of super module
  Double_t mSmodPar1;    //!<! y size of super module
  Double_t mSmodPar2;    //!<! z size of super module
  Double_t mInnerEdge;   //!<! Inner edge of DCAL super module
  Double_t mParEMOD[5];  //!<! parameters of EMCAL module (TRD1,2)

  ClassDefOverride(Detector, 1)
};
}
}
#endif
