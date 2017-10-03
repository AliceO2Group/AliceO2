// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author R+Preghenella - August 2017

#ifndef ALICEO2_DATAFORMATS_HEAVYIONINFO_H_
#define ALICEO2_DATAFORMATS_HEAVYIONINFO_H_

#include "SimulationDataFormat/GeneratorInfo.h"

namespace o2
{
namespace dataformats
{

/*****************************************************************/
/*****************************************************************/

class HeavyIonInfo : public GeneratorInfo
{

 public:
  /** default constructor **/
  HeavyIonInfo();
  /** copy constructor **/
  HeavyIonInfo(const HeavyIonInfo& rhs);
  /** operator= **/
  HeavyIonInfo& operator=(const HeavyIonInfo& rhs);
  /** destructor **/
  ~HeavyIonInfo() override;

  /** getters **/
  Int_t getNcollHard() const { return mNcollHard; };
  Int_t getNpartProj() const { return mNpartProj; };
  Int_t getNpartTarg() const { return mNpartTarg; };
  Int_t getNcoll() const { return mNcoll; };
  Int_t getNspecNeut() const { return mNspecNeut; };
  Int_t getNspecProt() const { return mNspecProt; };
  Double_t getImpactParameter() const { return mImpactParameter; };
  Double_t getEventPlaneAngle() const { return mEventPlaneAngle; };
  Double_t getEccentricity() const { return mEccentricity; };
  Double_t getSigmaNN() const { return mSigmaNN; };
  Double_t getCentrality() const { return mCentrality; };

  /** setters **/
  void setNcollHard(Int_t val) { mNcollHard = val; };
  void setNpartProj(Int_t val) { mNpartProj = val; };
  void setNpartTarg(Int_t val) { mNpartTarg = val; };
  void setNcoll(Int_t val) { mNcoll = val; };
  void setNspecNeut(Int_t val) { mNspecNeut = val; };
  void setNspecProt(Int_t val) { mNspecProt = val; };
  void setImpactParameter(Double_t val) { mImpactParameter = val; };
  void setEventPlaneAngle(Double_t val) { mEventPlaneAngle = val; };
  void setEccentricity(Double_t val) { mEccentricity = val; };
  void setSigmaNN(Double_t val) { mSigmaNN = val; };
  void setCentrality(Double_t val) { mCentrality = val; };

  /** methods **/
  void Print(Option_t* opt = "") const override;
  void Reset() override;

  /** statics **/
  static std::string keyName() { return "heavy-ion"; };

 protected:
  /** data members **/
  Int_t mNcollHard;          // Number of hard collisions
  Int_t mNpartProj;          // Number of participating nucleons in the projectile
  Int_t mNpartTarg;          // Number of participating nucleons in the target
  Int_t mNcoll;              // Number of collisions
  Int_t mNspecNeut;          // Number of spectator neutrons
  Int_t mNspecProt;          // Number of spectator protons
  Double_t mImpactParameter; // Impact parameter
  Double_t mEventPlaneAngle; // Event plane angle
  Double_t mEccentricity;    // Eccentricity
  Double_t mSigmaNN;         // Assumed nucleon-nucleon cross-section
  Double_t mCentrality;      // Centrality

  ClassDefOverride(HeavyIonInfo, 1);

}; /** class HeavyIonInfo **/

/*****************************************************************/
/*****************************************************************/

} /* namespace dataformats */
} /* namespace o2 */

#endif /* ALICEO2_DATAFORMATS_HEAVYIONINFO_H_ */
