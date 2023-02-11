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

/// \file GPUDisplayMagneticField.cxx
/// \author Piotr Nowakowski

#include "GPUDisplayMagneticField.h"
#include "GPULogging.h"

#include <fstream>
#include <random>
#include <cassert>

#ifndef GPUCA_NO_ROOT
#include <TGeoGlobalMagField.h>
#endif
#if !defined(GPUCA_NO_ROOT) && defined(GPUCA_O2_LIB)
#include "DetectorsBase/GeometryManager.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsBase/Propagator.h"
#endif

using namespace GPUCA_NAMESPACE::gpu;

GPUDisplayMagneticField::GPUDisplayMagneticField()
{
  initializeUniforms();
}

#ifdef GPUCA_O2_LIB
GPUDisplayMagneticField::GPUDisplayMagneticField(o2::field::MagneticField* field)
{
  initializeUniformsFromField(field);
}
#endif

void GPUDisplayMagneticField::generateSeedPoints(std::size_t count)
{
  std::mt19937 rng(0xDEADBEEF);

  // TODO: fetch these values from somewhere?
  std::uniform_int_distribution<int> generator_x(-512, 512);
  std::uniform_int_distribution<int> generator_y(-512, 512);
  std::uniform_int_distribution<int> generator_z(-455 - 1312, -455 + 1312);

  mFieldLineSeedPoints.clear();

  for (std::size_t i = 0; i < count; ++i) {
    mFieldLineSeedPoints.emplace_back(generator_x(rng), generator_y(rng), generator_z(rng));
  }
}

template <std::size_t MAX_DIM1_SEGMENTS, std::size_t MAX_DIM2_SEGMENTS, std::size_t MAX_DIM3_SEGMENTS>
std::tuple<std::size_t, std::size_t, std::size_t> loadSegments(std::ifstream& file, GPUDisplayMagneticField::SegmentsUniform<MAX_DIM1_SEGMENTS, MAX_DIM2_SEGMENTS, MAX_DIM3_SEGMENTS>& segments)
{
  file.read(reinterpret_cast<char*>(&segments.MinZ), sizeof(segments.MinZ));
  file.read(reinterpret_cast<char*>(&segments.MaxZ), sizeof(segments.MaxZ));
  file.read(reinterpret_cast<char*>(&segments.MultiplicativeFactor), sizeof(segments.MultiplicativeFactor));

  std::size_t NSegDim1, NSegDim2, NSegDim3;

  file.read(reinterpret_cast<char*>(&NSegDim1), sizeof(std::int32_t));
  assert(NSegDim1 <= MAX_DIM1_SEGMENTS);

  segments.ZSegments = NSegDim1;

  file.read(reinterpret_cast<char*>(segments.SegDim1), NSegDim1 * sizeof(segments.SegDim1[0]));

  file.read(reinterpret_cast<char*>(segments.BegSegDim2), NSegDim1 * sizeof(segments.BegSegDim2[0]));
  file.read(reinterpret_cast<char*>(segments.NSegDim2), NSegDim1 * sizeof(segments.NSegDim2[0]));

  file.read(reinterpret_cast<char*>(&NSegDim2), sizeof(std::int32_t));
  assert(NSegDim2 <= MAX_DIM2_SEGMENTS);

  file.read(reinterpret_cast<char*>(segments.SegDim2), NSegDim2 * sizeof(segments.SegDim2[0]));

  file.read(reinterpret_cast<char*>(segments.BegSegDim3), NSegDim2 * sizeof(segments.BegSegDim3[0]));
  file.read(reinterpret_cast<char*>(segments.NSegDim3), NSegDim2 * sizeof(segments.NSegDim3[0]));

  file.read(reinterpret_cast<char*>(&NSegDim3), sizeof(std::int32_t));
  assert(NSegDim3 <= MAX_DIM3_SEGMENTS);

  file.read(reinterpret_cast<char*>(segments.SegDim3), NSegDim3 * sizeof(segments.SegDim3[0]));
  file.read(reinterpret_cast<char*>(segments.SegID), NSegDim3 * sizeof(segments.SegID[0]));

  return std::make_tuple(NSegDim1, NSegDim2, NSegDim3);
}

template <std::size_t DIMENSIONS, std::size_t MAX_PARAMETERIZATIONS, std::size_t MAX_ROWS, std::size_t MAX_COLUMNS, std::size_t MAX_COEFFICIENTS>
std::tuple<std::size_t, std::size_t, std::size_t, std::size_t> loadParams(std::ifstream& file, GPUDisplayMagneticField::ParametrizationUniform<MAX_PARAMETERIZATIONS, MAX_ROWS, MAX_COLUMNS, MAX_COEFFICIENTS>& parametrizations)
{
  std::size_t NParams, NRows, NColumns, NCoefficients;
  file.read(reinterpret_cast<char*>(&NParams), sizeof(std::int32_t));
  assert(NParams <= (MAX_PARAMETERIZATIONS / DIMENSIONS));

  file.read(reinterpret_cast<char*>(parametrizations.BOffsets), DIMENSIONS * NParams * sizeof(parametrizations.BOffsets[0]));
  file.read(reinterpret_cast<char*>(parametrizations.BScales), DIMENSIONS * NParams * sizeof(parametrizations.BScales[0]));
  file.read(reinterpret_cast<char*>(parametrizations.BMin), DIMENSIONS * NParams * sizeof(parametrizations.BMin[0]));
  file.read(reinterpret_cast<char*>(parametrizations.BMax), DIMENSIONS * NParams * sizeof(parametrizations.BMax[0]));

  file.read(reinterpret_cast<char*>(parametrizations.NRows), DIMENSIONS * NParams * sizeof(parametrizations.NRows[0]));
  file.read(reinterpret_cast<char*>(parametrizations.ColsAtRowOffset), DIMENSIONS * NParams * sizeof(parametrizations.ColsAtRowOffset[0]));
  file.read(reinterpret_cast<char*>(parametrizations.CofsAtRowOffset), DIMENSIONS * NParams * sizeof(parametrizations.CofsAtRowOffset[0]));

  file.read(reinterpret_cast<char*>(&NRows), sizeof(std::int32_t));
  assert(NRows <= MAX_ROWS);

  file.read(reinterpret_cast<char*>(parametrizations.NColsAtRow), NRows * sizeof(parametrizations.NColsAtRow[0]));
  file.read(reinterpret_cast<char*>(parametrizations.CofsAtColOffset), NRows * sizeof(parametrizations.CofsAtColOffset[0]));

  file.read(reinterpret_cast<char*>(&NColumns), sizeof(std::int32_t));
  assert(NColumns <= MAX_COLUMNS);

  file.read(reinterpret_cast<char*>(parametrizations.NCofsAtCol), NColumns * sizeof(parametrizations.NCofsAtCol[0]));
  file.read(reinterpret_cast<char*>(parametrizations.AtColCoefOffset), NColumns * sizeof(parametrizations.AtColCoefOffset[0]));

  file.read(reinterpret_cast<char*>(&NCoefficients), sizeof(std::int32_t));
  assert(NCoefficients <= MAX_COEFFICIENTS);

  file.read(reinterpret_cast<char*>(parametrizations.Coeffs), NCoefficients * sizeof(parametrizations.Coeffs[0]));

  return std::make_tuple(NParams, NRows, NColumns, NCoefficients);
}

#ifndef GPUCA_O2_LIB
int GPUDisplayMagneticField::initializeUniforms()
{
  mSolenoidSegments = std::make_unique<SolenoidSegmentsUniform>();
  mDipoleSegments = std::make_unique<DipoleSegmentsUniform>();
  mSolenoidParameterization = std::make_unique<SolenoidParameterizationUniform>();
  mDipoleParameterization = std::make_unique<DipoleParameterizationUniform>();

  std::ifstream file;

  file.open("field.uniform", std::ifstream::binary);

  if (!file.good()) {
    GPUError("Cannot load field file \"field.uniform\" from the working directory.");
    return 1;
  }

  const auto [SolSegDim1, SolSegDim2, SolSegDim3] = loadSegments(file, *mSolenoidSegments);
  const auto [DipSegDim1, DipSegDim2, DipSegDim3] = loadSegments(file, *mDipoleSegments);
  const auto [SParams, SRows, SCols, SCoeffs] = loadParams<DIMENSIONS>(file, *mSolenoidParameterization);
  const auto [DParams, DRows, DCols, DCoeffs] = loadParams<DIMENSIONS>(file, *mDipoleParameterization);

  mSolSegDim1 = SolSegDim1;
  mSolSegDim2 = SolSegDim2;
  mSolSegDim3 = SolSegDim3;

  mSolParametrizations = SParams;
  mSolRows = SRows;
  mSolColumns = SCols;
  mSolCoefficients = SCoeffs;

  mDipSegDim1 = DipSegDim1;
  mDipSegDim2 = DipSegDim2;
  mDipSegDim3 = DipSegDim3;

  mDipParametrizations = DParams;
  mDipRows = DRows;
  mDipColumns = DCols;
  mDipCoefficients = DCoeffs;

  mRenderConstantsUniform = std::make_unique<RenderConstantsUniform>();

  return 0;
}
#endif

#if !defined(GPUCA_NO_ROOT) && defined(GPUCA_O2_LIB)
int GPUDisplayMagneticField::initializeUniforms()
{
  mRenderConstantsUniform = std::make_unique<RenderConstantsUniform>();

  auto* field = dynamic_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());

  if (!field) {
    const auto grp = o2::parameters::GRPObject::loadFrom();
    o2::base::GeometryManager::loadGeometry();
    o2::base::Propagator::initFieldFromGRP();
    field = dynamic_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
  }

  if (!field) {
    GPUError("Error loading magnetic field data");
    return 1;
  }

  return initializeUniformsFromField(field);
}

int GPUDisplayMagneticField::initializeUniformsFromField(o2::field::MagneticField* field)
{
  const auto chebMap = field->getMeasuredMap();

  const auto numberOfParameterizationSolenoid = chebMap->getNumberOfParametersSol();

  if (numberOfParameterizationSolenoid > MAX_SOLENOID_PARAMETERIZATIONS) {
    GPUError("Model has too many Solenoid parametrization regions to fit in the designated uniform");
    return 1;
  }

  const auto numberOfParameterizationDipole = chebMap->getNumberOfParametersDip();

  if (numberOfParameterizationDipole > MAX_DIPOLE_PARAMETERIZATIONS) {
    GPUError("Model has too many Dipole parametrization regions to fit in the designated uniform");
    return 1;
  }

  struct TableInfo {
    Float_t minZ;
    Float_t maxZ;
    Int_t numberOfDistinctDim1Segments;
    Int_t numberOfDistinctDim2Segments;
    Int_t numberOfDistinctDim3Segments;
    Float_t* coordinatesSegmentsDim1;
    Float_t* coordinatesSegmentsDim2;
    Float_t* coordinatesSegmentsDim3;
    Int_t* beginningOfSegmentsDim2;
    Int_t* beginningOfSegmentsDim3;
    Int_t* numberOfSegmentsDim2;
    Int_t* numberOfSegmentsDim3;
    Int_t* segmentId;

    ~TableInfo()
    {
      delete[] coordinatesSegmentsDim1;
      delete[] coordinatesSegmentsDim2;
      delete[] coordinatesSegmentsDim3;
      delete[] beginningOfSegmentsDim2;
      delete[] beginningOfSegmentsDim3;
      delete[] numberOfSegmentsDim2;
      delete[] numberOfSegmentsDim3;
      delete[] segmentId;
    }
  };

  TableInfo solenoidTableInfo, dipoleTableInfo;

  const auto parameterizationSolenoid = std::make_unique<TObjArray>();

  for (auto i = 0; i < numberOfParameterizationSolenoid; ++i) {
    parameterizationSolenoid->Add(chebMap->getParameterSolenoid(i));
  }

  chebMap->buildTable(numberOfParameterizationSolenoid, parameterizationSolenoid.get(), solenoidTableInfo.numberOfDistinctDim1Segments,
                      solenoidTableInfo.numberOfDistinctDim2Segments, solenoidTableInfo.numberOfDistinctDim3Segments, solenoidTableInfo.minZ, solenoidTableInfo.maxZ,
                      &solenoidTableInfo.coordinatesSegmentsDim1, &solenoidTableInfo.coordinatesSegmentsDim2, &solenoidTableInfo.coordinatesSegmentsDim3,
                      &solenoidTableInfo.beginningOfSegmentsDim2, &solenoidTableInfo.numberOfSegmentsDim2, &solenoidTableInfo.beginningOfSegmentsDim3,
                      &solenoidTableInfo.numberOfSegmentsDim3, &solenoidTableInfo.segmentId);

  LOGF(info, "Solenoid minZ: %f maxZ: %f", solenoidTableInfo.minZ, solenoidTableInfo.maxZ);
  LOGF(info, "Solenoid segment count: Z %d P %d R %d", solenoidTableInfo.numberOfDistinctDim1Segments, solenoidTableInfo.numberOfDistinctDim2Segments, solenoidTableInfo.numberOfDistinctDim3Segments);

  const auto parameterizationDipole = std::make_unique<TObjArray>();

  for (auto i = 0; i < numberOfParameterizationDipole; ++i) {
    parameterizationDipole->Add(chebMap->getParameterDipole(i));
  }

  chebMap->buildTable(numberOfParameterizationDipole, parameterizationDipole.get(), dipoleTableInfo.numberOfDistinctDim1Segments,
                      dipoleTableInfo.numberOfDistinctDim2Segments, dipoleTableInfo.numberOfDistinctDim3Segments, dipoleTableInfo.minZ, dipoleTableInfo.maxZ,
                      &dipoleTableInfo.coordinatesSegmentsDim1, &dipoleTableInfo.coordinatesSegmentsDim2, &dipoleTableInfo.coordinatesSegmentsDim3,
                      &dipoleTableInfo.beginningOfSegmentsDim2, &dipoleTableInfo.numberOfSegmentsDim2, &dipoleTableInfo.beginningOfSegmentsDim3,
                      &dipoleTableInfo.numberOfSegmentsDim3, &dipoleTableInfo.segmentId);

  LOGF(info, "Dipole minZ: %f maxZ: %f", dipoleTableInfo.minZ, dipoleTableInfo.maxZ);
  LOGF(info, "Dipole segment count: Z %d Y %d X %d", dipoleTableInfo.numberOfDistinctDim1Segments, dipoleTableInfo.numberOfDistinctDim2Segments, dipoleTableInfo.numberOfDistinctDim3Segments);

  const auto getParameterSolenoid = [chebMap](Int_t i) { return chebMap->getParameterSolenoid(i); };
  const auto getParameterDipole = [chebMap](Int_t i) { return chebMap->getParameterDipole(i); };

  auto countArraySizes = [](int numberOfParametrization, auto& getParameter) {
    int TotalRows = 0, TotalColumns = 0, TotalCoefficients = 0;
    UShort_t MaxChebyshevOrder = 0;

    for (auto i = 0; i < numberOfParametrization; ++i) {
      const auto param = getParameter(i);

      for (auto j = 0; j < DIMENSIONS; ++j) {
        const auto calc = param->getChebyshevCalc(j);

        const auto NRows = calc->getNumberOfRows();
        TotalRows += NRows;

        const auto colsAtRow = calc->getNumberOfColumnsAtRow();

        for (auto row = 0; row < NRows; ++row) {
          const auto NCols = colsAtRow[row];
          TotalColumns += NCols;

          const auto col0 = calc->getColAtRowBg()[row];

          for (auto col = 0; col < NCols; ++col) {
            const auto ncoffs = calc->getCoefficientBound2D0()[col0 + col];
            MaxChebyshevOrder = std::max(MaxChebyshevOrder, ncoffs);
          }
        }

        TotalCoefficients += calc->getNumberOfCoefficients();
      }
    }

    return std::make_tuple(TotalRows, TotalColumns, TotalCoefficients, MaxChebyshevOrder);
  };

  const auto [TotalRowsSolenoid, TotalColumnsSolenoid, TotalCoefficientsSolenoid, MaxChebyshevOrderSolenoid] = countArraySizes(numberOfParameterizationSolenoid, getParameterSolenoid);
  const auto [TotalRowsDipole, TotalColumnsDipole, TotalCoefficientsDipole, MaxChebyshevOrderDipole] = countArraySizes(numberOfParameterizationDipole, getParameterDipole);

  LOGF(info, "Solenoid Params: %d Rows %d Columns %d Coefficients %d MaxChebOrder: %d", numberOfParameterizationSolenoid, TotalRowsSolenoid, TotalColumnsSolenoid, TotalCoefficientsSolenoid, MaxChebyshevOrderSolenoid);
  LOGF(info, "Dipole Params: %d Rows %d Columns %d Coefficients %d MaxChebOrder: %d", numberOfParameterizationDipole, TotalRowsDipole, TotalColumnsDipole, TotalCoefficientsDipole, MaxChebyshevOrderDipole);

  bool spaceOk = true;
  std::string reason;

  if (solenoidTableInfo.numberOfDistinctDim1Segments > MAX_SOLENOID_Z_SEGMENTS) {
    spaceOk = false;
    reason = "Solenoid Z segment";
  } else if (solenoidTableInfo.numberOfDistinctDim2Segments > MAX_SOLENOID_P_SEGMENTS) {
    spaceOk = false;
    reason = "Solenoid P segment";
  } else if (solenoidTableInfo.numberOfDistinctDim3Segments > MAX_SOLENOID_R_SEGMENTS) {
    spaceOk = false;
    reason = "Solenoid R segment";
  } else if (dipoleTableInfo.numberOfDistinctDim1Segments > MAX_DIPOLE_Z_SEGMENTS) {
    spaceOk = false;
    reason = "Dipole Z Segment";
  } else if (dipoleTableInfo.numberOfDistinctDim2Segments > MAX_DIPOLE_Y_SEGMENTS) {
    spaceOk = false;
    reason = "Dipole Y Segment";
  } else if (dipoleTableInfo.numberOfDistinctDim3Segments > MAX_DIPOLE_X_SEGMENTS) {
    spaceOk = false;
    reason = "Dipole X Segment";
  } else if (TotalRowsSolenoid > MAX_SOLENOID_ROWS) {
    spaceOk = false;
    reason = "Solenoid Rows";
  } else if (TotalColumnsSolenoid > MAX_SOLENOID_COLUMNS) {
    spaceOk = false;
    reason = "Solenoid Columns";
  } else if (TotalCoefficientsSolenoid > MAX_SOLENOID_COEFFICIENTS) {
    spaceOk = false;
    reason = "Solenoid Coefficients";
  } else if (TotalRowsDipole > MAX_DIPOLE_ROWS) {
    spaceOk = false;
    reason = "Dipole Rows";
  } else if (TotalColumnsDipole > MAX_DIPOLE_COLUMNS) {
    spaceOk = false;
    reason = "Dipole Columns";
  } else if (TotalCoefficientsDipole > MAX_DIPOLE_COEFFICIENTS) {
    spaceOk = false;
    reason = "Dipole Coefficients";
  } else if (std::max(MaxChebyshevOrderSolenoid, MaxChebyshevOrderDipole) > MAX_CHEBYSHEV_ORDER) {
    spaceOk = false;
    reason = "Chebyshev Polynomial order";
  }

  if (!spaceOk) {
    GPUError("Model has too large %s to fit in the designated uniform", reason);
    return 1;
  }

  auto loadSegments = [](const auto& info, auto& segments) {
    std::copy(info.coordinatesSegmentsDim1, info.coordinatesSegmentsDim1 + info.numberOfDistinctDim1Segments, segments.SegDim1);

    std::copy(info.beginningOfSegmentsDim2, info.beginningOfSegmentsDim2 + info.numberOfDistinctDim1Segments, segments.BegSegDim2);
    std::copy(info.numberOfSegmentsDim2, info.numberOfSegmentsDim2 + info.numberOfDistinctDim1Segments, segments.NSegDim2);

    std::copy(info.coordinatesSegmentsDim2, info.coordinatesSegmentsDim2 + info.numberOfDistinctDim2Segments, segments.SegDim2);

    std::copy(info.beginningOfSegmentsDim3, info.beginningOfSegmentsDim3 + info.numberOfDistinctDim2Segments, segments.BegSegDim3);
    std::copy(info.numberOfSegmentsDim3, info.numberOfSegmentsDim3 + info.numberOfDistinctDim2Segments, segments.NSegDim3);
    std::copy(info.coordinatesSegmentsDim3, info.coordinatesSegmentsDim3 + info.numberOfDistinctDim3Segments, segments.SegDim3);

    std::copy(info.segmentId, info.segmentId + info.numberOfDistinctDim3Segments, segments.SegID);
  };

  mSolenoidSegments = std::make_unique<SolenoidSegmentsUniform>();
  loadSegments(solenoidTableInfo, *mSolenoidSegments);

  mDipoleSegments = std::make_unique<DipoleSegmentsUniform>();
  loadSegments(dipoleTableInfo, *mDipoleSegments);

  auto initParametrization = [](int numberOfParametrization, auto& getParameter, auto& parametrizationUniform) {
    int ColsAtRowOffset = 0, CofsAtRowOffset = 0, CofsAtColOffset = 0, Coeffs = 0;

    for (auto i = 0; i < numberOfParametrization; ++i) {
      const auto param = getParameter(i);
      std::copy(param->getBoundaryMappingOffset(), param->getBoundaryMappingOffset() + DIMENSIONS, &parametrizationUniform->BOffsets[i * DIMENSIONS]);
      std::copy(param->getBoundaryMappingScale(), param->getBoundaryMappingScale() + DIMENSIONS, &parametrizationUniform->BScales[i * DIMENSIONS]);
      std::copy(param->getBoundMin(), param->getBoundMin() + DIMENSIONS, &parametrizationUniform->BMin[i * DIMENSIONS]);
      std::copy(param->getBoundMax(), param->getBoundMax() + DIMENSIONS, &parametrizationUniform->BMax[i * DIMENSIONS]);

      for (auto j = 0; j < DIMENSIONS; ++j) {
        const auto calc = param->getChebyshevCalc(j);

        const auto NRows = calc->getNumberOfRows();

        parametrizationUniform->NRows[i * DIMENSIONS + j] = NRows;
        parametrizationUniform->ColsAtRowOffset[i * DIMENSIONS + j] = ColsAtRowOffset;
        parametrizationUniform->CofsAtRowOffset[i * DIMENSIONS + j] = CofsAtRowOffset;

        const auto colsAtRow = calc->getNumberOfColumnsAtRow();

        for (auto row = 0; row < NRows; ++row) {
          const auto NCols = colsAtRow[row];
          parametrizationUniform->NColsAtRow[ColsAtRowOffset + row] = NCols;
          parametrizationUniform->CofsAtColOffset[ColsAtRowOffset + row] = CofsAtColOffset;

          const auto col0 = calc->getColAtRowBg()[row];

          for (auto col = 0; col < NCols; ++col) {
            const auto ncoffs = calc->getCoefficientBound2D0()[col0 + col];
            const auto offset = calc->getCoefficientBound2D1()[col0 + col];

            parametrizationUniform->NCofsAtCol[CofsAtColOffset + col] = ncoffs;
            parametrizationUniform->AtColCoefOffset[CofsAtColOffset + col] = offset;
          }
          CofsAtColOffset += NCols;
        }

        ColsAtRowOffset += NRows;
        CofsAtRowOffset += calc->getNumberOfCoefficients();

        std::copy(calc->getCoefficients(), calc->getCoefficients() + calc->getNumberOfCoefficients(), &parametrizationUniform->Coeffs[Coeffs]);

        Coeffs += calc->getNumberOfCoefficients();
      }
    }

    return std::make_tuple(ColsAtRowOffset, CofsAtColOffset, Coeffs);
  };

  mSolenoidParameterization = std::make_unique<SolenoidParameterizationUniform>();
  mDipoleParameterization = std::make_unique<DipoleParameterizationUniform>();

  const auto [SRows, SCols, SCoeffs] = initParametrization(numberOfParameterizationSolenoid, getParameterSolenoid, mSolenoidParameterization);
  const auto [DRows, DCols, DCoeffs] = initParametrization(numberOfParameterizationDipole, getParameterDipole, mDipoleParameterization);

  assert(TotalRowsSolenoid == SRows);
  assert(TotalColumnsSolenoid == SCols);
  assert(TotalCoefficientsSolenoid == SCoeffs);
  assert(TotalRowsDipole == DRows);
  assert(TotalColumnsDipole == DCols);
  assert(TotalCoefficientsDipole == DCoeffs);

  mSolenoidSegments->MinZ = solenoidTableInfo.minZ;
  mSolenoidSegments->MaxZ = solenoidTableInfo.maxZ;
  mSolenoidSegments->MultiplicativeFactor = field->getFactorSolenoid();
  mSolenoidSegments->ZSegments = solenoidTableInfo.numberOfDistinctDim1Segments;

  mDipoleSegments->MinZ = dipoleTableInfo.minZ;
  mDipoleSegments->MaxZ = dipoleTableInfo.maxZ;
  mDipoleSegments->MultiplicativeFactor = field->getFactorDipole();
  mDipoleSegments->ZSegments = dipoleTableInfo.numberOfDistinctDim1Segments;

  mSolSegDim1 = solenoidTableInfo.numberOfDistinctDim1Segments;
  mSolSegDim2 = solenoidTableInfo.numberOfDistinctDim2Segments;
  mSolSegDim3 = solenoidTableInfo.numberOfDistinctDim3Segments;

  mSolParametrizations = numberOfParameterizationSolenoid;
  mSolRows = TotalRowsSolenoid;
  mSolColumns = TotalColumnsSolenoid;
  mSolCoefficients = TotalCoefficientsSolenoid;

  mDipSegDim1 = dipoleTableInfo.numberOfDistinctDim1Segments;
  mDipSegDim2 = dipoleTableInfo.numberOfDistinctDim2Segments;
  mDipSegDim3 = dipoleTableInfo.numberOfDistinctDim3Segments;

  mDipParametrizations = numberOfParameterizationDipole;
  mDipRows = TotalRowsDipole;
  mDipColumns = TotalColumnsDipole;
  mDipCoefficients = TotalCoefficientsDipole;

  return 0;
}
#endif
