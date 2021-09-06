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

#ifndef ALICEO2_EMCAL_ELMBMEASUREMENT_H_
#define ALICEO2_EMCAL_ELMBMEASUREMENT_H_

#include <vector>
#include <tuple>
#include <Rtypes.h>

#include "EMCALCalib/ElmbData.h"

namespace o2
{

namespace emcal
{

//typedef std::tuple <int, float, float, float, float> Sensor_t; //{Npoints, mean, rms, min, max}

class ElmbMeasurement
{

 public:
  ElmbMeasurement() = default;
  ~ElmbMeasurement() = default;

  void init();
  void process();
  void reset();

  void addMeasurement(int iPT, double val) { values[iPT].push_back(val); }

  std::vector<Sensor_t> getData() { return mELMBdata; }
  std::vector<std::vector<double>> getValues() { return values; }

 private:
  std::vector<std::vector<double>> values; ///<container with measured values
  std::vector<float> values_prev;          ///< last measurement per sensor
  std::vector<Sensor_t> mELMBdata;         //

  ClassDefNV(ElmbMeasurement, 1);
};

} // namespace emcal

} // namespace o2
#endif
