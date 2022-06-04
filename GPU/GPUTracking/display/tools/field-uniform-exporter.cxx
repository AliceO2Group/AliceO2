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

/// @file   field-uniform-exporter.cxx
/// @author Piotr Nowakowski

#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <Field/MagneticField.h>
#include <memory>

#include "../GPUDisplayMagneticField.h"

namespace bpo = boost::program_options;
using namespace GPUCA_NAMESPACE::gpu;

template <std::size_t MAX_DIM1_SEGMENTS, std::size_t MAX_DIM2_SEGMENTS, std::size_t MAX_DIM3_SEGMENTS>
void saveSegments(std::ofstream& file, int NSegDim1, int NSegDim2, int NSegDim3, GPUDisplayMagneticField::SegmentsUniform<MAX_DIM1_SEGMENTS, MAX_DIM2_SEGMENTS, MAX_DIM3_SEGMENTS>& segments)
{
  file.write(reinterpret_cast<char*>(&segments.MinZ), sizeof(segments.MinZ));
  file.write(reinterpret_cast<char*>(&segments.MaxZ), sizeof(segments.MaxZ));
  file.write(reinterpret_cast<char*>(&segments.MultiplicativeFactor), sizeof(segments.MultiplicativeFactor));

  file.write(reinterpret_cast<char*>(&NSegDim1), sizeof(std::int32_t));

  file.write(reinterpret_cast<char*>(segments.SegDim1), NSegDim1 * sizeof(segments.SegDim1[0]));

  file.write(reinterpret_cast<char*>(segments.BegSegDim2), NSegDim1 * sizeof(segments.BegSegDim2[0]));
  file.write(reinterpret_cast<char*>(segments.NSegDim2), NSegDim1 * sizeof(segments.NSegDim2[0]));

  file.write(reinterpret_cast<char*>(&NSegDim2), sizeof(std::int32_t));

  file.write(reinterpret_cast<char*>(segments.SegDim2), NSegDim2 * sizeof(segments.SegDim2[0]));

  file.write(reinterpret_cast<char*>(segments.BegSegDim3), NSegDim2 * sizeof(segments.BegSegDim3[0]));
  file.write(reinterpret_cast<char*>(segments.NSegDim3), NSegDim2 * sizeof(segments.NSegDim3[0]));

  file.write(reinterpret_cast<char*>(&NSegDim3), sizeof(std::int32_t));

  file.write(reinterpret_cast<char*>(segments.SegDim3), NSegDim3 * sizeof(segments.SegDim3[0]));
  file.write(reinterpret_cast<char*>(segments.SegID), NSegDim3 * sizeof(segments.SegID[0]));
}

template <std::size_t DIMENSIONS, std::size_t MAX_PARAMETERIZATIONS, std::size_t MAX_ROWS, std::size_t MAX_COLUMNS, std::size_t MAX_COEFFICIENTS>
void saveParams(std::ofstream& file, int numberOfParametrization, int Nrows, int Ncolums, int Ncoeffs, GPUDisplayMagneticField::ParametrizationUniform<MAX_PARAMETERIZATIONS, MAX_ROWS, MAX_COLUMNS, MAX_COEFFICIENTS>& param)
{
  file.write(reinterpret_cast<char*>(&numberOfParametrization), sizeof(std::int32_t));

  file.write(reinterpret_cast<char*>(param.BOffsets), DIMENSIONS * numberOfParametrization * sizeof(float));
  file.write(reinterpret_cast<char*>(param.BScales), DIMENSIONS * numberOfParametrization * sizeof(float));
  file.write(reinterpret_cast<char*>(param.BMin), DIMENSIONS * numberOfParametrization * sizeof(float));
  file.write(reinterpret_cast<char*>(param.BMax), DIMENSIONS * numberOfParametrization * sizeof(float));

  file.write(reinterpret_cast<char*>(param.NRows), DIMENSIONS * numberOfParametrization * sizeof(float));
  file.write(reinterpret_cast<char*>(param.ColsAtRowOffset), DIMENSIONS * numberOfParametrization * sizeof(float));
  file.write(reinterpret_cast<char*>(param.CofsAtRowOffset), DIMENSIONS * numberOfParametrization * sizeof(float));

  file.write(reinterpret_cast<char*>(&Nrows), sizeof(std::int32_t));

  file.write(reinterpret_cast<char*>(param.NColsAtRow), Nrows * sizeof(int));
  file.write(reinterpret_cast<char*>(param.CofsAtColOffset), Nrows * sizeof(int));

  file.write(reinterpret_cast<char*>(&Ncolums), sizeof(int));

  file.write(reinterpret_cast<char*>(param.NCofsAtCol), Ncolums * sizeof(int));
  file.write(reinterpret_cast<char*>(param.AtColCoefOffset), Ncolums * sizeof(int));

  file.write(reinterpret_cast<char*>(&Ncoeffs), sizeof(int));

  file.write(reinterpret_cast<char*>(param.Coeffs), Ncoeffs * sizeof(float));
}

int main(int argc, char** argv)
{
  bpo::options_description options("Field Uniform exporter options");

  float l3Cur, diCur, beamenergy;
  bool uniform;
  std::string beamtype, filename;

  options.add_options()(
    "help,h", "produce help message")(
    "l3current,l", bpo::value<decltype(l3Cur)>()->default_value(30000.f), "current of the L3 magnet")(
    "dipolecurrent,d", bpo::value<decltype(diCur)>()->default_value(6000.f), "current of the Dipole magnet")(
    "beamenergy,b", bpo::value<decltype(beamenergy)>()->default_value(7000.f), "beam energy")(
    "beamtype,t", bpo::value<decltype(beamtype)>()->default_value("pp"), "beam type")(
    "uniform,u", bpo::value<decltype(uniform)>()->default_value(false), "uniform")(
    "filename,f", bpo::value<decltype(beamtype)>()->default_value("field.uniform"), "output filename");

  using namespace bpo::command_line_style;
  auto style = (allow_short | short_allow_adjacent | short_allow_next | allow_long | long_allow_adjacent | long_allow_next | allow_sticky | allow_dash_for_short);
  bpo::variables_map varmap;
  try {
    bpo::store(
      bpo::command_line_parser(argc, argv)
        .options(options)
        .style(style)
        .run(),
      varmap);
  } catch (std::exception const& e) {
    LOGP(error, "error parsing options of {}: {}", argv[0], e.what());
    return 1;
  }

  if (varmap.count("help")) {
    LOG(info) << options << std::endl;
    return 0;
  }

  l3Cur = varmap["l3current"].as<decltype(l3Cur)>();
  diCur = varmap["dipolecurrent"].as<decltype(diCur)>();
  beamenergy = varmap["beamenergy"].as<decltype(beamenergy)>();
  beamtype = varmap["beamtype"].as<decltype(beamtype)>();
  uniform = varmap["uniform"].as<decltype(uniform)>();
  filename = varmap["filename"].as<decltype(filename)>();

  const auto field = std::unique_ptr<o2::field::MagneticField>(o2::field::MagneticField::createFieldMap(-l3Cur, -diCur, 0, uniform, beamenergy, beamtype.c_str()));
  const auto gpufield = std::make_unique<GPUDisplayMagneticField>(field.get());

  auto file = std::ofstream(filename, std::ios::out | std::ios::binary);

  saveSegments(file, gpufield->mSolSegDim1, gpufield->mSolSegDim2, gpufield->mSolSegDim3, *gpufield->mSolenoidSegments);
  saveSegments(file, gpufield->mDipSegDim1, gpufield->mDipSegDim2, gpufield->mDipSegDim3, *gpufield->mDipoleSegments);
  saveParams<o2::gpu::GPUDisplayMagneticField::DIMENSIONS>(file, gpufield->mSolParametrizations, gpufield->mSolRows, gpufield->mSolColumns, gpufield->mSolCoefficients, *gpufield->mSolenoidParameterization);
  saveParams<o2::gpu::GPUDisplayMagneticField::DIMENSIONS>(file, gpufield->mDipParametrizations, gpufield->mDipRows, gpufield->mDipColumns, gpufield->mDipCoefficients, *gpufield->mDipoleParameterization);

  file.close();

  return 0;
}
