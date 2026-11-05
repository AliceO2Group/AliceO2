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

/// \file GPUDisplayMagneticField.h
/// \author Piotr Nowakowski

#ifndef GPUDISPLAYMAGNETICFIELD_H
#define GPUDISPLAYMAGNETICFIELD_H

#include "GPUCommonDef.h"
#include "GPUDisplayMagneticFieldConstants.h" // Shared with the field shader
#include <memory>
#include <vector>

#ifndef GPUCA_STANDALONE
#include <Field/MagneticField.h>
#endif

namespace o2::gpu
{
class GPUDisplayMagneticField
{
 public:
  GPUDisplayMagneticField();
#ifndef GPUCA_STANDALONE
  GPUDisplayMagneticField(o2::field::MagneticField* field);
#endif

  static constexpr std::size_t DIMENSIONS = GPUCA_DISPLAY_FIELD_DIMENSIONS;
  static constexpr std::size_t MAX_SOLENOID_Z_SEGMENTS = GPUCA_DISPLAY_FIELD_MAX_SOLENOID_Z_SEGMENTS;
  static constexpr std::size_t MAX_SOLENOID_P_SEGMENTS = GPUCA_DISPLAY_FIELD_MAX_SOLENOID_P_SEGMENTS;
  static constexpr std::size_t MAX_SOLENOID_R_SEGMENTS = GPUCA_DISPLAY_FIELD_MAX_SOLENOID_R_SEGMENTS;
  static constexpr std::size_t MAX_DIPOLE_Z_SEGMENTS = GPUCA_DISPLAY_FIELD_MAX_DIPOLE_Z_SEGMENTS;
  static constexpr std::size_t MAX_DIPOLE_Y_SEGMENTS = GPUCA_DISPLAY_FIELD_MAX_DIPOLE_Y_SEGMENTS;
  static constexpr std::size_t MAX_DIPOLE_X_SEGMENTS = GPUCA_DISPLAY_FIELD_MAX_DIPOLE_X_SEGMENTS;
  static constexpr std::size_t MAX_SOLENOID_PARAMETERIZATIONS = GPUCA_DISPLAY_FIELD_MAX_SOLENOID_PARAMETERIZATIONS;
  static constexpr std::size_t MAX_SOLENOID_ROWS = GPUCA_DISPLAY_FIELD_MAX_SOLENOID_ROWS;
  static constexpr std::size_t MAX_SOLENOID_COLUMNS = GPUCA_DISPLAY_FIELD_MAX_SOLENOID_COLUMNS;
  static constexpr std::size_t MAX_SOLENOID_COEFFICIENTS = GPUCA_DISPLAY_FIELD_MAX_SOLENOID_COEFFICIENTS;
  static constexpr std::size_t MAX_DIPOLE_PARAMETERIZATIONS = GPUCA_DISPLAY_FIELD_MAX_DIPOLE_PARAMETERIZATIONS;
  static constexpr std::size_t MAX_DIPOLE_ROWS = GPUCA_DISPLAY_FIELD_MAX_DIPOLE_ROWS;
  static constexpr std::size_t MAX_DIPOLE_COLUMNS = GPUCA_DISPLAY_FIELD_MAX_DIPOLE_COLUMNS;
  static constexpr std::size_t MAX_DIPOLE_COEFFICIENTS = GPUCA_DISPLAY_FIELD_MAX_DIPOLE_COEFFICIENTS;
  static constexpr std::size_t MAX_CHEBYSHEV_ORDER = GPUCA_DISPLAY_FIELD_MAX_CHEBYSHEV_ORDER;

  struct RenderConstantsUniform {
    uint32_t StepCount;
    float StepSize;
  };

  template <std::size_t MAX_DIM1_SEGMENTS, std::size_t MAX_DIM2_SEGMENTS, std::size_t MAX_DIM3_SEGMENTS>
  struct SegmentsUniform {
    float MinZ;
    float MaxZ;
    float MultiplicativeFactor;

    int32_t ZSegments;

    float SegDim1[MAX_DIM1_SEGMENTS];

    int32_t BegSegDim2[MAX_DIM1_SEGMENTS];
    int32_t NSegDim2[MAX_DIM1_SEGMENTS];

    float SegDim2[MAX_DIM2_SEGMENTS];

    int32_t BegSegDim3[MAX_DIM2_SEGMENTS];
    int32_t NSegDim3[MAX_DIM2_SEGMENTS];

    float SegDim3[MAX_DIM3_SEGMENTS];

    int32_t SegID[MAX_DIM3_SEGMENTS];
  };

  template <std::size_t MAX_PARAMETERIZATIONS, std::size_t MAX_ROWS, std::size_t MAX_COLUMNS, std::size_t MAX_COEFFICIENTS>
  struct ParametrizationUniform {
    float BOffsets[MAX_PARAMETERIZATIONS];
    float BScales[MAX_PARAMETERIZATIONS];
    float BMin[MAX_PARAMETERIZATIONS];
    float BMax[MAX_PARAMETERIZATIONS];

    int32_t NRows[MAX_PARAMETERIZATIONS];
    int32_t ColsAtRowOffset[MAX_PARAMETERIZATIONS];
    int32_t CofsAtRowOffset[MAX_PARAMETERIZATIONS];

    int32_t NColsAtRow[MAX_ROWS];
    int32_t CofsAtColOffset[MAX_ROWS];

    int32_t NCofsAtCol[MAX_COLUMNS];
    int32_t AtColCoefOffset[MAX_COLUMNS];

    float Coeffs[MAX_COEFFICIENTS];
  };

  using SolenoidSegmentsUniform = SegmentsUniform<MAX_SOLENOID_Z_SEGMENTS, MAX_SOLENOID_P_SEGMENTS, MAX_SOLENOID_R_SEGMENTS>;
  using SolenoidParameterizationUniform = ParametrizationUniform<DIMENSIONS * MAX_SOLENOID_PARAMETERIZATIONS, MAX_SOLENOID_ROWS, MAX_SOLENOID_COLUMNS, MAX_SOLENOID_COEFFICIENTS>;

  using DipoleSegmentsUniform = SegmentsUniform<MAX_DIPOLE_Z_SEGMENTS, MAX_DIPOLE_Y_SEGMENTS, MAX_DIPOLE_X_SEGMENTS>;
  using DipoleParameterizationUniform = ParametrizationUniform<DIMENSIONS * MAX_DIPOLE_PARAMETERIZATIONS, MAX_DIPOLE_ROWS, MAX_DIPOLE_COLUMNS, MAX_DIPOLE_COEFFICIENTS>;

  // TODO: what to do with this?
  struct vtx {
    float x, y, z;
    vtx(float a, float b, float c) : x(a), y(b), z(c) {}
  };

  int32_t initializeUniforms();
#ifndef GPUCA_STANDALONE
  int32_t initializeUniformsFromField(o2::field::MagneticField* field);
#endif
  void generateSeedPoints(std::size_t count);

  std::size_t mSolSegDim1;
  std::size_t mSolSegDim2;
  std::size_t mSolSegDim3;

  std::size_t mDipSegDim1;
  std::size_t mDipSegDim2;
  std::size_t mDipSegDim3;

  std::size_t mSolParametrizations;
  std::size_t mSolRows;
  std::size_t mSolColumns;
  std::size_t mSolCoefficients;

  std::size_t mDipParametrizations;
  std::size_t mDipRows;
  std::size_t mDipColumns;
  std::size_t mDipCoefficients;

  bool mValid = false; // Field data loaded, buffers below filled
  std::unique_ptr<RenderConstantsUniform> mRenderConstantsUniform;
  std::unique_ptr<SolenoidSegmentsUniform> mSolenoidSegments;
  std::unique_ptr<DipoleSegmentsUniform> mDipoleSegments;
  std::unique_ptr<SolenoidParameterizationUniform> mSolenoidParameterization;
  std::unique_ptr<DipoleParameterizationUniform> mDipoleParameterization;
  std::vector<vtx> mFieldLineSeedPoints;
};
} // namespace o2::gpu

#endif // GPUDISPLAYMAGNETICFIELD_H
