// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_ITS3_ALIGNMENT_MISALIGNMENTUTILS_H
#define O2_ITS3_ALIGNMENT_MISALIGNMENTUTILS_H

#include <array>
#include <cstddef>
#include <map>
#include <string>
#include <utility>

#include "ITS3Align/AlignmentMath.h"
#include "MathUtils/LegendrePols.h"

namespace o2::its3::align
{

struct InextensionalMisalignment {
  std::map<int, std::array<double, 4>> modes; // n -> (a_n, b_n, c_n, d_n)
  double alpha{0.};
  double beta{0.};
};

struct SensorMisalignment {
  o2::math_utils::Legendre2DPolynominal legendre;
  bool hasLegendre{false};
  InextensionalMisalignment inextensional;
  bool hasInextensional{false};

  bool empty() const noexcept { return !hasLegendre && !hasInextensional; }
};

struct MisalignmentModel {
  static constexpr std::size_t NSensors = 6;
  std::array<SensorMisalignment, NSensors> sensors{};

  bool empty() const noexcept;
  const SensorMisalignment& operator[](std::size_t idx) const { return sensors[idx]; }
  SensorMisalignment& operator[](std::size_t idx) { return sensors[idx]; }
};

struct MisalignmentFrame {
  int sensorID{-1};
  int layerID{-1};
  double x{0.};     // tracking-frame X / nominal radius at the measurement
  double alpha{0.}; // tracking-frame alpha
  double z{0.};     // tracking-frame measurement z
};

struct MisalignmentShift {
  double dy{0.};
  double dz{0.};
  bool accepted{true};

  MisalignmentShift& operator+=(const MisalignmentShift& other)
  {
    dy += other.dy;
    dz += other.dz;
    accepted = accepted && other.accepted;
    return *this;
  }
};

MisalignmentModel loadMisalignmentModel(const std::string& jsonPath);
MisalignmentShift evaluateLegendreShift(const SensorMisalignment& sensor, const MisalignmentFrame& frame, const TrackSlopes& slopes);
MisalignmentShift evaluateInextensionalShift(const SensorMisalignment& sensor, const MisalignmentFrame& frame, const TrackSlopes& slopes);

} // namespace o2::its3::align

#endif
