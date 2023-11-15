// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ALICE3MagneticField.h
/// \brief A simple magnetic field class for ALICE3 R&D
/// \author sandro.wenzel@cern.ch

#ifndef ALICEO2_FIELD_ALICE3MAGNETICFIELD_H_
#define ALICEO2_FIELD_ALICE3MAGNETICFIELD_H_

#include "FairField.h" // for FairField
#include "Rtypes.h"    // for ClassDef

namespace o2
{
namespace field
{

/// A simple magnetic field class for ALICE3 R&D. Can easily
/// be used in Virtual Monte Carlo simulations.
class ALICE3MagneticField : public FairField
{
 public:
  ALICE3MagneticField() : FairField()
  {
    fType = 2;
    init();
  }

  ~ALICE3MagneticField() override = default;

  /// X component, avoid using since slow
  Double_t GetBx(Double_t x, Double_t y, Double_t z) override
  {
    double xyz[3] = {x, y, z}, b[3];
    ALICE3MagneticField::Field(xyz, b);
    return b[0];
  }

  /// Y component, avoid using since slow
  Double_t GetBy(Double_t x, Double_t y, Double_t z) override
  {
    double xyz[3] = {x, y, z}, b[3];
    ALICE3MagneticField::Field(xyz, b);
    return b[1];
  }

  /// Z component
  Double_t GetBz(Double_t x, Double_t y, Double_t z) override
  {
    double xyz[3] = {x, y, z}, b[3];
    ALICE3MagneticField::Field(xyz, b);
    return b[2];
  }

  /// Method to calculate the field at point xyz
  /// Main interface from TVirtualMagField used in simulation
  void Field(const Double_t* __restrict__ point, Double_t* __restrict__ bField) override;

 private:
  // defining a function type, that could be initialized during runtime from a ROOT macro
  // to ease fast R&D prototyping
  typedef std::function<void(const double* __restrict__, double* __restrict__)> FieldEvalFcn;
  FieldEvalFcn mJITFieldFunction; //!

  void init();
  void initJITFieldFunction();

  //  ClassDefOverride(o2::field::ALICE3MagneticField, 1)
};

} // end namespace field
} // end namespace o2

#endif
