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

#ifndef ALICEO2_TPC_ParameterGEM_H_
#define ALICEO2_TPC_ParameterGEM_H_

#include <array>

namespace o2 {
namespace TPC {

/// \class ParameterGEM

class ParameterGEM{
  public:
    static ParameterGEM& defaultInstance() {
      static ParameterGEM param;
      param.setDefaultValues();
      return param;
    }

    /// Constructor
    ParameterGEM();

    /// Destructor
    ~ParameterGEM() = default;

    /// Set the default values
    void setDefaultValues();

    /// Set absolute gain for the stack
    /// \param absGain1 Absolute gain in GEM 1
    /// \param absGain2 Absolute gain in GEM 2
    /// \param absGain3 Absolute gain in GEM 3
    /// \param absGain4 Absolute gain in GEM 4
    void setAbsoluteGain(float absGain1, float absGain2, float absGain3, float absGain4);

    /// Set absolute gain for a single GEM in the stack
    /// \param absGain Absolute gain
    /// \param GEM GEM of interest in the stack (1 - 4)
    void setAbsoluteGain(float absGain, int gem) { mAbsoluteGain[gem-1] = absGain; }

    /// Set collection efficiency for the stack
    /// \param collEff1 Collection efficiency in GEM 1
    /// \param collEff2 Collection efficiency in GEM 2
    /// \param collEff3 Collection efficiency in GEM 3
    /// \param collEff4 Collection efficiency in GEM 4
    void setCollectionEfficiency(float collEff1, float collEff2, float collEff3, float collEff4);

    /// Set collection efficiency for a single GEM inthe stack
    /// \param collEff Collection efficiency
    /// \param GEM GEM of interest in the stack (1 - 4)
    void setCollectionEfficiency(float collEff, int gem) { mCollectionEfficiency[gem-1] = collEff; }

    /// Set extraction efficiency for the stack
    /// \param extrEff1 Extraction efficiency in GEM 1
    /// \param extrEff2 Extraction efficiency in GEM 2
    /// \param extrEff3 Extraction efficiency in GEM 3
    /// \param extrEff4 Extraction efficiency in GEM 4
    void setExtractionEfficiency(float extrEff1, float extrEff2, float extrEff3, float extrEff4);

    /// Set extraction efficiency for a single GEM inthe stack
    /// \param extrEff Extraction efficiency
    /// \param GEM GEM of interest in the stack (1 - 4)
    void setExtractionEfficiency(float extrEff, int gem) { mExtractionEfficiency[gem-1] = extrEff; }


    /// Get the effective gain of a given GEM in the stack
    /// \param GEM GEM of interest in the stack (1 - 4)
    /// \return Effective gain of a given GEM in the stack
    float getEffectiveGain(int gem) const { return mCollectionEfficiency[gem-1]*mAbsoluteGain[gem-1]*mExtractionEfficiency[gem-1]; }

    /// Get the absolute gain of a given GEM in the stack
    /// \param GEM GEM of interest in the stack (1 - 4)
    /// \return Absolute gain of a given GEM in the stack
    float getAbsoluteGain(int gem) const { return mAbsoluteGain[gem-1]; }

    /// Get the collection efficiency of a given GEM in the stack
    /// \param GEM GEM of interest in the stack (1 - 4)
    /// \return Collection efficiency of a given GEM in the stack
    float getCollectionEfficiency(int gem) const { return mCollectionEfficiency[gem-1]; }

    /// Get the extraction efficiency of a given GEM in the stack
    /// \param GEM GEM of interest in the stack (1 - 4)
    /// \return Extraction efficiency of a given GEM in the stack
    float getExtractionEfficiency(int gem) const { return mExtractionEfficiency[gem-1]; }

  private:

    /// \todo Remove hard-coded number of GEMs in a stack
    std::array<float, 4> mAbsoluteGain;         ///< Absolute gain
    std::array<float, 4> mCollectionEfficiency; ///< Collection efficiency
    std::array<float, 4> mExtractionEfficiency; ///< Extraction efficiency
  };

inline
void ParameterGEM::setAbsoluteGain(float absGain1, float absGain2, float absGain3, float absGain4)
{
  mAbsoluteGain = {{absGain1, absGain2, absGain3, absGain4}};
}

inline
void ParameterGEM::setCollectionEfficiency(float collEff1, float collEff2, float collEff3, float collEff4)
{
  mCollectionEfficiency = {{collEff1, collEff2, collEff3, collEff4}};
}

inline
void ParameterGEM::setExtractionEfficiency(float extrEff1, float extrEff2, float extrEff3, float extrEff4)
{
  mExtractionEfficiency = {{extrEff1, extrEff2, extrEff3, extrEff4}};
}

}
}

#endif // ALICEO2_TPC_ParameterGEM_H_
