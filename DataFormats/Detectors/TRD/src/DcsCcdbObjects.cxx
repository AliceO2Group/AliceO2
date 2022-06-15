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

/// \file DcsCcdbObjects.cxx
/// \brief Objects which are created from DCS DPs and stored in the CCDB
/// \author Ole Schmidt, ole.schmidt@cern.ch

#include "DataFormatsTRD/DcsCcdbObjects.h"
#include <fairlogger/Logger.h>
#include <string>

using namespace o2::trd;

void TRDDCSMinMaxMeanInfo::print() const
{
  std::time_t tStart = firstTS / 1e3;
  std::string tStartStr = std::asctime(std::localtime(&tStart));
  tStartStr.erase(tStartStr.length() - 1); // remove '\n', since it is added by FairLogger anyways
  std::time_t tEnd = lastTS / 1e3;
  std::string tEndStr = std::asctime(std::localtime(&tEnd));
  tEndStr.erase(tEndStr.length() - 1);
  LOG(info) << "Raw time interval from " << firstTS << " to " << lastTS << ". In local time:";
  LOG(info) << tStartStr;
  LOG(info) << tEndStr;
  LOG(info) << "Min value: " << minValue;
  LOG(info) << "Max value: " << maxValue;
  LOG(info) << "Mean value: " << meanValue;
  LOG(info) << "Number of points added: " << nPoints;
}

void TRDDCSMinMaxMeanInfo::addPoint(float value, uint64_t ts)
{
  if (nPoints == 0) {
    firstTS = ts;
    lastTS = ts;
    minValue = value;
    maxValue = value;
    meanValue = value;
  } else {
    if (value < minValue) {
      minValue = value;
    }
    if (value > maxValue) {
      maxValue = value;
    }
    // I am not sure whether the DPs arrive always ordered in time
    if (ts > lastTS) {
      lastTS = ts;
    }
    if (ts < firstTS) {
      firstTS = ts;
    }
    meanValue += (value - meanValue) / (nPoints + 1);
  }
  ++nPoints;
}
