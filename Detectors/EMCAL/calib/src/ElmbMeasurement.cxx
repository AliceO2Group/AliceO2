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

#include <iostream>

#include "TMath.h"
#include "EMCALCalib/ElmbMeasurement.h"

using namespace o2::emcal;

void ElmbMeasurement::init()
{
  values.resize(NElmbSensors);
  values_prev.resize(NElmbSensors, -1);
  mELMBdata.resize(NElmbSensors);
}

void ElmbMeasurement::reset()
{

  for (int iPT = 0; iPT < NElmbSensors; iPT++) {
    values[iPT].clear();
  }
  mELMBdata.clear();

  mELMBdata.resize(NElmbSensors); // check why the capacity is 0 after clear()!!!
}

void ElmbMeasurement::process()
{
  int Npoints = 0;
  float val_last = 0;
  double mean = 0;
  double mean2 = 0;
  double rms = 0.;
  double max = 0;
  double min = 1000;

  for (int iPT = 0; iPT < NElmbSensors; iPT++) {
    Npoints = 0;
    mean = 0;
    mean2 = 0;
    max = 0;
    min = 1000;
    rms = 0;
    val_last = values_prev[iPT];

    //    std::cout<< iPT << " : ";
    for (auto vPT : values[iPT]) {
      //            std::cout << vPT << ", ";
      Npoints++;
      mean += vPT;
      mean2 += vPT * vPT;
      if (max < vPT) {
        max = vPT;
      }
      if (min > vPT) {
        min = vPT;
      }
      val_last = (float)vPT;
    }
    //    std::cout << std::endl;

    if (Npoints == 0) {
      mean = (double)values_prev[iPT];
    } else if (Npoints > 1) {
      mean2 /= Npoints;
      mean /= Npoints;
      rms = mean2 - mean * mean;
      rms /= (Npoints - 1);
      rms = TMath::Sqrt(rms);
    }

    values_prev[iPT] = val_last;
    mELMBdata[iPT] = std::make_tuple(Npoints, (float)mean, (float)rms, (float)min, (float)max);
  }
}
