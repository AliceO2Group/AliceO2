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
// implement the efficiency models for the collection and the
// extraction efficiency.

#ifndef ALICEO2_TPC_ParameterGEM_H_
#define ALICEO2_TPC_ParameterGEM_H_

#include <array>

namespace o2
{
namespace TPC
{

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

  /// Set GEM geometry for the stack (1 standard pitch, 2 medium pitch, 3 large pitch)
  /// \param geom1 Geometry for GEM 1
  /// \param geom2 Geometry for GEM 2
  /// \param geom3 Geometry for GEM 3
  /// \param geom4 Geometry for GEM 4
  void setGeometry(int geom1, int geom2, int geom3, int geom4);

  /// Set GEM geometry for a single GEM in the stack (1 standard pitch, 2 medium pitch, 3 large pitch)
  /// \param geom Geometry for GEM
  /// \param GEM GEM of interest in the stack (1 - 4)
  void setGeometry(int geom, int gem)
  {
    mGeometry[gem - 1] = geom;
  }

  /// Set potential for the stack (in Volts)
  /// \param pot1 Potential for GEM 1
  /// \param pot2 Potential for GEM 2
  /// \param pot3 Potential for GEM 3
  /// \param pot4 Potential for GEM 4
  void setPotential(float pot1, float pot2, float pot3, float pot4);

  /// Set potential for a single GEM in the stack (in Volts)
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

  /// Set collection efficiency for a single GEM inthe stack
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

  /// Set extraction efficiency for a single GEM inthe stack
  /// \param extrEff Extraction efficiency
  /// \param GEM GEM of interest in the stack (1 - 4)
  void setExtractionEfficiency(float extrEff, int gem)
  {
    mExtractionEfficiency[gem - 1] = extrEff;
  }

  /// Get the geometry type of a given GEM in the stack
  /// \param GEM GEM of interest in the stack (1 - 4)
  /// \return Geometry type (0 standard, 1 medium, 2 large)
  int getGeometry(int gem) const
  {
    return mGeometry[gem - 1];
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

 private:
  /// \todo Remove hard-coded number of GEMs in a stack
  std::array<int, 4> mGeometry;               ///< GEM geometry (1 standard, 2 medium, 3 large)
  std::array<float, 4> mPotential;            ///< Potential (in Volts)
  std::array<float, 5> mElectricField;        ///< Electric field configuration (in kV/cm)
  std::array<float, 4> mAbsoluteGain;         ///< Absolute gain
  std::array<float, 4> mCollectionEfficiency; ///< Collection efficiency
  std::array<float, 4> mExtractionEfficiency; ///< Extraction efficiency
};

inline void ParameterGEM::setGeometry(int geom1, int geom2, int geom3, int geom4)
{
  mGeometry = { { geom1, geom2, geom3, geom4 } };
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
}
}

#endif // ALICEO2_TPC_ParameterGEM_H_
