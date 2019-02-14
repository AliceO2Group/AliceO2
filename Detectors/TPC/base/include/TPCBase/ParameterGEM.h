// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ParameterGEM.h
/// \brief Definition of the parameter class for the GEM stack
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

// Remark: This file has been modified by Viktor Ratza in order to
// implement the electron efficiency models for the collection and the
// extraction efficiency.

#ifndef ALICEO2_TPC_ParameterGEM_H_
#define ALICEO2_TPC_ParameterGEM_H_

#include <array>
#include <cmath>

namespace o2
{
namespace TPC
{

enum class AmplificationMode : char {
  FullMode = 0,      ///< Full 4-GEM simulation of all efficiencies etc.
  EffectiveMode = 1, ///< Effective amplification mode using one polya distribution only
};

/// \class ParameterGEM

class ParameterGEM
{
 public:
  static ParameterGEM& defaultInstance()
  {
    static ParameterGEM param;
    return param;
  }

  /// Constructor
  ParameterGEM();

  /// Destructor
  ~ParameterGEM() = default;

  /// Set GEM geometry for the stack (0 standard pitch, 1 medium pitch, 2 large pitch)
  /// \param geom1 Geometry for GEM 1
  /// \param geom2 Geometry for GEM 2
  /// \param geom3 Geometry for GEM 3
  /// \param geom4 Geometry for GEM 4
  void setGeometry(int geom1, int geom2, int geom3, int geom4);

  /// Set GEM geometry for a single GEM in the stack (0 standard pitch, 1 medium pitch, 2 large pitch)
  /// \param geom Geometry for GEM
  /// \param GEM GEM of interest in the stack (1 - 4)
  void setGeometry(int geom, int gem)
  {
    mGeometry[gem - 1] = geom;
  }

  /// Set distances between cathode-GEM1, between GEMs and GEM4-anode
  /// \param distance1 Distance cathode-GEM1 (drift region) in cm
  /// \param distance2 Distance GEM1-GEM2 (ET1) in cm
  /// \param distance3 Distance GEM2-GEM3 (ET2) in cm
  /// \param distance4 Distance GEM3-GEM4 (ET3) in cm
  /// \param distance5 Distance GEM4-anode (induction region) in cm
  void setDistance(float distance1, float distance2, float distance3, float distance4, float distance5);

  /// Set the distance for a single region in the stack
  /// \param distance Distance for the region in cm
  /// \param region Region of interest in the stack (1 Drift, 2 ET1, 3 ET2, 4 ET3, 5 Induction)
  void setDistance(float distance, int region)
  {
    mDistance[region - 1] = distance;
  }

  /// Set potential for the stack (in volt)
  /// \param pot1 Potential for GEM 1
  /// \param pot2 Potential for GEM 2
  /// \param pot3 Potential for GEM 3
  /// \param pot4 Potential for GEM 4
  void setPotential(float pot1, float pot2, float pot3, float pot4);

  /// Set potential for a single GEM in the stack (in volt)
  /// \param pot Potential for GEM
  /// \param GEM GEM of interest in the stack (1 - 4)
  void setPotential(float pot, int gem)
  {
    mPotential[gem - 1] = pot;
  }

  /// Set electric field configuration for the stack (in kV/cm)
  /// \param elecField1 Electric field in drift region
  /// \param elecField2 Electric field between GEM1 and GEM2 (ET1)
  /// \param elecField3 Electric field between GEM2 and GEM3 (ET2)
  /// \param elecField4 Electric field between GEM3 and GEM4 (ET3)
  /// \param elecField5 Electric field in induction region
  void setElectricField(float elecField1, float elecField2, float elecField3, float elecField4, float elecField5);

  /// Set electric field for a single region in the stack (in kV/cm)
  /// \param elecField Electric field for the region
  /// \param region Region of interest in the stack (1 Drift, 2 ET1, 3 ET2, 4 ET3, 5 Induction)
  void setElectricField(float elecField, int region)
  {
    mElectricField[region - 1] = elecField;
  }

  /// Set absolute gain for the stack
  /// \param absGain1 Absolute gain in GEM 1
  /// \param absGain2 Absolute gain in GEM 2
  /// \param absGain3 Absolute gain in GEM 3
  /// \param absGain4 Absolute gain in GEM 4
  void setAbsoluteGain(float absGain1, float absGain2, float absGain3, float absGain4);

  /// Set absolute gain for a single GEM in the stack
  /// \param absGain Absolute gain
  /// \param GEM GEM of interest in the stack (1 - 4)
  void setAbsoluteGain(float absGain, int gem)
  {
    mAbsoluteGain[gem - 1] = absGain;
  }

  /// Set collection efficiency for the stack
  /// \param collEff1 Collection efficiency in GEM 1
  /// \param collEff2 Collection efficiency in GEM 2
  /// \param collEff3 Collection efficiency in GEM 3
  /// \param collEff4 Collection efficiency in GEM 4
  void setCollectionEfficiency(float collEff1, float collEff2, float collEff3, float collEff4);

  /// Set collection efficiency for a single GEM in the stack
  /// \param collEff Collection efficiency
  /// \param GEM GEM of interest in the stack (1 - 4)
  void setCollectionEfficiency(float collEff, int gem)
  {
    mCollectionEfficiency[gem - 1] = collEff;
  }

  /// Set extraction efficiency for the stack
  /// \param extrEff1 Extraction efficiency in GEM 1
  /// \param extrEff2 Extraction efficiency in GEM 2
  /// \param extrEff3 Extraction efficiency in GEM 3
  /// \param extrEff4 Extraction efficiency in GEM 4
  void setExtractionEfficiency(float extrEff1, float extrEff2, float extrEff3, float extrEff4);

  /// Set extraction efficiency for a single GEM in the stack
  /// \param extrEff Extraction efficiency
  /// \param GEM GEM of interest in the stack (1 - 4)
  void setExtractionEfficiency(float extrEff, int gem)
  {
    mExtractionEfficiency[gem - 1] = extrEff;
  }

  /// Set the total gain of the stack for the EffectiveMode
  /// \param totGain Total gain of the stack for the EffectiveMode
  void setTotalGainStack(float totGain) { mTotalGainStack = totGain; }

  /// Set the variable steering the energy resolution of the full stack for the EffectiveMode
  /// \param kappa Variable steering the energy resolution of the full stack for the EffectiveMode
  void setKappaStack(float kappa) { mKappaStack = kappa; }

  /// Set the variable steering the single electron efficiency  of the full stack for the EffectiveMode
  /// \param eff Variable steering the single electron efficiency of the full stack for the EffectiveMode
  void setEfficiencyStack(float eff) { mEfficiencyStack = eff; }

  /// Get the amplification mode to be used
  /// \param mode Amplification mode to be used
  void setAmplificationMode(AmplificationMode mode) { mAmplificationMode = mode; }

  /// Get the geometry type of a given GEM in the stack
  /// \param GEM GEM of interest in the stack (1 - 4)
  /// \return Geometry type (0 standard, 1 medium, 2 large)
  int getGeometry(int gem) const
  {
    return mGeometry[gem - 1];
  }

  /// Get the distance between cathode-GEM1, between GEMs or GEM4-anode
  /// \param region Region of interest in the stack (1 Drift, 2 ET1, 3 ET2, 4 ET3, 5 Induction)
  /// \return Distance of region in cm
  float getDistance(int region) const
  {
    return mDistance[region - 1];
  }

  /// Get the electric potential of a given GEM in the stack
  /// \param GEM GEM of interest in the stack (1 - 4)
  /// \return Electric potential of GEM in Volts
  float getPotential(int gem) const
  {
    return mPotential[gem - 1];
  }

  /// Get the electric field configuration for a given GEM stack
  /// \param region Region of interest in the stack (1 Drift, 2 ET1, 3 ET2, 4 ET3, 5 Induction)
  /// \return Electric field in kV/cm
  float getElectricField(int region) const
  {
    return mElectricField[region - 1];
  }

  /// Get the effective gain of a given GEM in the stack
  /// \param GEM GEM of interest in the stack (1 - 4)
  /// \return Effective gain of a given GEM in the stack
  float getEffectiveGain(int gem) const
  {
    return mCollectionEfficiency[gem - 1] * mAbsoluteGain[gem - 1] * mExtractionEfficiency[gem - 1];
  }

  /// Get the absolute gain of a given GEM in the stack
  /// \param GEM GEM of interest in the stack (1 - 4)
  /// \return Absolute gain of a given GEM in the stack
  float getAbsoluteGain(int gem) const
  {
    return mAbsoluteGain[gem - 1];
  }

  /// Get the collection efficiency of a given GEM in the stack
  /// \param GEM GEM of interest in the stack (1 - 4)
  /// \return Collection efficiency of a given GEM in the stack
  float getCollectionEfficiency(int gem) const
  {
    return mCollectionEfficiency[gem - 1];
  }

  /// Get the extraction efficiency of a given GEM in the stack
  /// \param GEM GEM of interest in the stack (1 - 4)
  /// \return Extraction efficiency of a given GEM in the stack
  float getExtractionEfficiency(int gem) const
  {
    return mExtractionEfficiency[gem - 1];
  }

  /// Get the total gain of the stack for the EffectiveMode
  /// \return Total gain of the stack for the EffectiveMode
  float getTotalGainStack() const { return mTotalGainStack; }

  /// Get the variable steering the energy resolution of the full stack for the EffectiveMode
  /// \return Variable steering the energy resolution of the full stack for the EffectiveMode
  float getKappaStack() const { return mKappaStack; }

  /// Get the variable steering the single electron efficiency  of the full stack for the EffectiveMode
  /// \return Variable steering the single electron efficiency of the full stack for the EffectiveMode
  float getEfficiencyStack() const { return mEfficiencyStack; }

  /// Get the amplification mode to be used
  /// \return Amplification mode to be used
  AmplificationMode getAmplificationMode() const { return mAmplificationMode; }

 private:
  /// \todo Remove hard-coded number of GEMs in a stack
  std::array<int, 4> mGeometry;               ///< GEM geometry (0 standard, 1 medium, 2 large)
  std::array<float, 5> mDistance;             ///< Distances between cathode/anode and stages (in cm)
  std::array<float, 4> mPotential;            ///< Potential (in Volts)
  std::array<float, 5> mElectricField;        ///< Electric field configuration (in kV/cm)
  std::array<float, 4> mAbsoluteGain;         ///< Absolute gain
  std::array<float, 4> mCollectionEfficiency; ///< Collection efficiency
  std::array<float, 4> mExtractionEfficiency; ///< Extraction efficiency
  float mTotalGainStack;                      ///< Total gain of the stack for the EffectiveMode
  float mKappaStack;                          ///< Variable steering the energy resolution of the full stack for the EffectiveMode
  float mEfficiencyStack;                     ///< Variable steering the single electron efficiency of the full stack for the EffectiveMode
  AmplificationMode mAmplificationMode;       ///< Amplification mode [FullMode / EffectiveMode]
};

inline void ParameterGEM::setGeometry(int geom1, int geom2, int geom3, int geom4)
{
  mGeometry = { { geom1, geom2, geom3, geom4 } };
}

inline void ParameterGEM::setDistance(float distance1, float distance2, float distance3, float distance4,
                                      float distance5)
{
  mDistance = { { distance1, distance2, distance3, distance4, distance5 } };
}

inline void ParameterGEM::setPotential(float pot1, float pot2, float pot3, float pot4)
{
  mPotential = { { pot1, pot2, pot3, pot4 } };
}

inline void ParameterGEM::setElectricField(float elecField1, float elecField2, float elecField3, float elecField4,
                                           float elecField5)
{
  mElectricField = { { elecField1, elecField2, elecField3, elecField4, elecField5 } };
}

inline void ParameterGEM::setAbsoluteGain(float absGain1, float absGain2, float absGain3, float absGain4)
{
  mAbsoluteGain = { { absGain1, absGain2, absGain3, absGain4 } };
}

inline void ParameterGEM::setCollectionEfficiency(float collEff1, float collEff2, float collEff3, float collEff4)
{
  mCollectionEfficiency = { { collEff1, collEff2, collEff3, collEff4 } };
}

inline void ParameterGEM::setExtractionEfficiency(float extrEff1, float extrEff2, float extrEff3, float extrEff4)
{
  mExtractionEfficiency = { { extrEff1, extrEff2, extrEff3, extrEff4 } };
}
} // namespace TPC
} // namespace o2

#endif // ALICEO2_TPC_ParameterGEM_H_
