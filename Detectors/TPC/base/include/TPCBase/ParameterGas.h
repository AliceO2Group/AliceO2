// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ParameterGas.h
/// \brief Definition of the parameter class for the detector gas
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_ParameterGas_H_
#define ALICEO2_TPC_ParameterGas_H_

#include <array>

namespace o2
{
namespace TPC
{

/// \class ParameterGas

class ParameterGas
{
 public:
  static ParameterGas& defaultInstance()
  {
    static ParameterGas param;
    return param;
  }

  /// Constructor
  ParameterGas();

  /// Destructor
  ~ParameterGas() = default;

  /// Set effective ionization potential
  /// \param wion Effective ionization potential [GeV]
  void setWion(float wion) { mWion = wion; }

  /// Set first ionization potential
  /// \param ipot First ionization potential [GeV]
  void setIpot(float ipot) { mIpot = ipot; }

  /// Set maximum energy we are able to deposit [GeV]
  void setEend(float eend) { mEend = eend; }

  /// Set the exponent of the energy loss
  void setExp(float exp) { mExp = exp; }

  /// Set attachment coefficient
  /// \param attcoeff Attachment coefficient [1/m]
  void setAttachmentCoefficient(float attcoeff) { mAttCoeff = attcoeff; }

  /// Set oxygen content
  /// \param oxygenCont Oxygen content [1E6 ppm]
  void setOxygenContent(float oxygenContent) { mOxyCont = oxygenContent; }

  /// Set drift velocity
  /// \param vdrift Drift velocify [cm/us]
  void setVdrift(float vdrift) { mDriftV = vdrift; }

  /// Set Sigma over Mu
  /// \param som Sigma over Mu
  void setSigmaOverMu(float som) { mSigmaOverMu = som; }

  /// Set transverse diffusion
  /// \param diffT Transverse diffusion [sqrt(cm)]
  void setDiffT(float diffT) { mDiffT = diffT; }

  /// Set longitudinal diffusion
  /// \param diffL Longitudinal diffusion [sqrt(cm)]
  void setDiffL(float diffL) { mDiffL = diffL; }

  /// Set number of primary electrons per cm and MIP
  /// \param nprim Number of primary electrons per cm and MIP [1/cm]
  void setNprim(float nprim) { mNprim = nprim; }

  /// Set scale factor to tune WION for GEANT4
  /// \param scale Scale factor to tune WION for GEANT4
  void setScaleG4(float scale) { mScaleFactorG4 = scale; }

  /// Set parameter for smearing the number of ionizations (nel) using GEANT4
  /// \param param Parameter for smearing the number of ionizations (nel) using GEANT4
  void setFanoFactorG4(float param) { mFanoFactorG4 = param; }

  /// Set Bethe-Bloch parameters
  /// \param paramX Bethe-Bloch parameter
  void setBetheBlochParam(float param1, float param2, float param3, float param4, float param5);

  /// Get effective ionization potential
  /// \return Effective ionization potential [GeV]
  float getWion() const { return mWion; }

  /// Get first ionization potential
  /// \return First ionization potential [GeV]
  float getIpot() const { return mIpot; }

  /// Get maximum energy we are able to deposit [GeV]
  /// \return Maximum energy we are able to deposit [GeV]
  float getEend() const { return mEend; }

  /// Get the exponent of the energy loss
  /// \return Exponent of the energy loss
  float getExp() const { return mExp; }

  /// Get attachment coefficient
  /// \return Attachment coefficient [1/m]
  float getAttachmentCoefficient() const { return mAttCoeff; }

  /// Get oxygen content
  /// \return Oxygen content [1E6 ppm]
  float getOxygenContent() const { return mOxyCont; }

  /// Get drift velocity
  /// \return Drift velocify [cm/us]
  float getVdrift() const { return mDriftV; }

  /// Get Sigma over Mu
  /// \return Sigma over Mu
  float getSigmaOverMu() const { return mSigmaOverMu; }

  /// Get transverse diffusion
  /// \return Transverse diffusion [sqrt(cm)]
  float getDiffT() const { return mDiffT; }

  /// Get longitudinal diffusion
  /// \return Longitudinal diffusion [sqrt(cm)]
  float getDiffL() const { return mDiffL; }

  /// Get number of primary electrons per cm and MIP
  /// \return Number of primary electrons per cm and MIP [1/cm]
  float getNprim() const { return mNprim; }

  /// Get scale factor to tune WION for GEANT4
  /// \return Scale factor to tune WION for GEANT4
  float getScaleG4() const { return mScaleFactorG4; }

  /// Get parameter for smearing the number of ionizations (nel) using GEANT4
  /// \return Parameter for smearing the number of ionizations (nel) using GEANT4
  float getFanoFactorG4() const { return mFanoFactorG4; }

  /// Get Bethe-Bloch parameters
  /// \return Bethe-Bloch parameter
  float getBetheBlochParam(int param) const { return mBetheBlochParam[param]; }

 private:
  float mWion;          ///< Effective ionization potential [GeV]
  float mIpot;          ///< First ionization potential [GeV]
  float mEend;          ///< Maximum allowed energy loss [GeV]
  float mExp;           ///< Exponent of the energy loss
  float mAttCoeff;      ///< Attachement coefficient [1/m]
  float mOxyCont;       ///< Oxygen content [1E6 ppm]
  float mDriftV;        ///< Drift velocity [cm/us]
  float mSigmaOverMu;   ///< Sigma over mu, gives deviation from exponential gain fluctuations
  float mDiffT;         ///< Transverse diffusion [sqrt(cm)]
  float mDiffL;         ///< Longitudinal diffusion [sqrt(cm)]
  float mNprim;         ///< Number of primary electrons per MIP and cm [1/cm]
  float mScaleFactorG4; ///< Scale factor to tune WION for GEANT4
  float mFanoFactorG4;  ///< Parameter for smearing the number of ionizations (nel) using GEANT4

  std::array<float, 5> mBetheBlochParam; ///< Bethe-Bloch parameters
};

inline void ParameterGas::setBetheBlochParam(float param1, float param2, float param3, float param4, float param5)
{
  mBetheBlochParam = { { param1, param2, param3, param4, param5 } };
}
}
}

#endif // ALICEO2_TPC_ParameterGas_H_
