// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ParameterDetector.h
/// \brief Definition of the parameter class for the detector
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_ParameterDetector_H_
#define ALICEO2_TPC_ParameterDetector_H_

#include <array>

namespace o2
{
namespace TPC
{

/// \class ParameterDetector

class ParameterDetector
{
 public:
  static ParameterDetector& defaultInstance()
  {
    static ParameterDetector param;
    param.setDefaultValues();
    return param;
  }

  /// Constructor
  ParameterDetector();

  /// Destructor
  ~ParameterDetector() = default;

  /// Set the default values
  void setDefaultValues();

  /// Set the TPC length
  /// \param tpclength TPC length [cm]
  void setTPClength(float tpclength) { mTPClength = tpclength; }

  /// Set the pad capacitance
  /// \param cpad Pad capacitance [pF]
  void setPadCapacitance(float cpad) { mPadCapacitance = cpad; }

  /// Get the TPC length
  /// \return TPC length [cm]
  float getTPClength() const { return mTPClength; }

  /// Get the pad capacitance
  /// \return Pad capacitance [pF]
  float getPadCapacitance() const { return mPadCapacitance; }

 private:
  float mTPClength;      ///< Length of the TPC [cm]
  float mPadCapacitance; ///< Capacitance of a single pad [pF]
};
}
}

#endif // ALICEO2_TPC_ParameterDetector_H_
