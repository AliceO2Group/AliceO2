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

#ifndef ALICEO2_EMCAL_ELMBDATA_H_
#define ALICEO2_EMCAL_ELMBDATA_H_

#include <vector>
#include <tuple>
#include <Rtypes.h>

namespace o2
{
namespace emcal
{

const int NElmbSensors = 180;
typedef std::tuple<int, float, float, float, float> Sensor_t; //{Npoints, mean, rms, min, max}

class ElmbData
{

 public:
  ElmbData() = default;
  ~ElmbData() = default;

  void setData(std::vector<Sensor_t> data) { mELMB = data; }
  void setSensor(int iSensor, Sensor_t data) { mELMB[iSensor] = data; }
  void setSensor(int iSensor, int Npoints, float mean, float rms, float min, float max)
  {
    mELMB[iSensor] = std::make_tuple(Npoints, mean, rms, min, max);
  }

  std::vector<Sensor_t> getData() { return mELMB; }
  Sensor_t getSensor(short iSensor) { return mELMB[iSensor]; }
  int getNpoints(short iSensor) { return std::get<0>(mELMB[iSensor]); }
  float getMean(short iSensor) { return std::get<1>(mELMB[iSensor]); }
  float getRMS(short iSensor) { return std::get<2>(mELMB[iSensor]); }
  float getMin(short iSensor) { return std::get<3>(mELMB[iSensor]); }
  float getMax(short iSensor) { return std::get<4>(mELMB[iSensor]); }

 private:
  std::vector<Sensor_t> mELMB; ///< data container

  ClassDefNV(ElmbData, 1);
};

} // namespace emcal

} // namespace o2

#endif