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
#include <string>

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
  int getNcollHard() const { return mNcollHard; };
  int getNpartProj() const { return mNpartProj; };
  int getNpartTarg() const { return mNpartTarg; };
  int getNcoll() const { return mNcoll; };
  int getNspecNeut() const { return mNspecNeut; };
  int getNspecProt() const { return mNspecProt; };
  double getImpactParameter() const { return mImpactParameter; };
  double getEventPlaneAngle() const { return mEventPlaneAngle; };
  double getEccentricity() const { return mEccentricity; };
  double getSigmaNN() const { return mSigmaNN; };
  double getCentrality() const { return mCentrality; };

  /** setters **/
  void setNcollHard(int val) { mNcollHard = val; };
  void setNpartProj(int val) { mNpartProj = val; };
  void setNpartTarg(int val) { mNpartTarg = val; };
  void setNcoll(int val) { mNcoll = val; };
  void setNspecNeut(int val) { mNspecNeut = val; };
  void setNspecProt(int val) { mNspecProt = val; };
  void setImpactParameter(double val) { mImpactParameter = val; };
  void setEventPlaneAngle(double val) { mEventPlaneAngle = val; };
  void setEccentricity(double val) { mEccentricity = val; };
  void setSigmaNN(double val) { mSigmaNN = val; };
  void setCentrality(double val) { mCentrality = val; };

  /** methods **/
  void print() const override;
  void reset() override;

  /** statics **/
  static std::string keyName() { return "heavy-ion"; };

 protected:
  /** data members **/
  int mNcollHard;          // Number of hard collisions
  int mNpartProj;          // Number of participating nucleons in the projectile
  int mNpartTarg;          // Number of participating nucleons in the target
  int mNcoll;              // Number of collisions
  int mNspecNeut;          // Number of spectator neutrons
  int mNspecProt;          // Number of spectator protons
  double mImpactParameter; // Impact parameter
  double mEventPlaneAngle; // Event plane angle
  double mEccentricity;    // Eccentricity
  double mSigmaNN;         // Assumed nucleon-nucleon cross-section
  double mCentrality;      // Centrality

}; /** class HeavyIonInfo **/

/*****************************************************************/
/*****************************************************************/

} /* namespace dataformats */
} /* namespace o2 */

#endif /* ALICEO2_DATAFORMATS_HEAVYIONINFO_H_ */
