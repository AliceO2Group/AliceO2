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

#ifndef MEAN_VERTEX_DATA_H_
#define MEAN_VERTEX_DATA_H_

#include "DetectorsCalibration/TimeSlot.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include <array>
#include <deque>
#include <gsl/span>

namespace o2
{
namespace calibration
{

struct MeanVertexData {

  using PVertex = o2::dataformats::PrimaryVertex;
  int entries = 0;
  std::vector<std::array<float, 3>> histoVtx{0};
  bool mVerbose = false;

  MeanVertexData() = default;

  ~MeanVertexData()
  {
    histoVtx.clear();
  }

  MeanVertexData(MeanVertexData&& other) = default;
  MeanVertexData(const MeanVertexData& other) = default;
  MeanVertexData& operator=(MeanVertexData& other) = default;
  MeanVertexData& operator=(MeanVertexData&& other) = default;

  //_____________________________________________

  size_t getEntries() const { return entries; }
  void print() const;
  void fill(const gsl::span<const PVertex> data);
  void merge(const MeanVertexData* prev);
  void subtract(const MeanVertexData* prev);
  void useVerboseMode(bool flag) { mVerbose = flag; }

  ClassDefNV(MeanVertexData, 1);
};

} // end namespace calibration
} // end namespace o2

#endif /* MEAN_VERTEX_DATA_H_ */
