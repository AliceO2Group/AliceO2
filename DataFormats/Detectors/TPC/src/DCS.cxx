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

/// @file   DCS.cxx
/// @author Jens Wiechula
/// @brief  DCS data point data formats

#include <limits>

#include "DataFormatsTPC/DCS.h"
#include "TLinearFitter.h"
#include "TTree.h"

using namespace o2::tpc::dcs;

//==============================================================================
//
//
const std::unordered_map<std::string, int> Temperature::SensorNameMap = {
  {"TPC_PT_351_TEMPERATURE", 0},
  {"TPC_PT_376_TEMPERATURE", 1},
  {"TPC_PT_415_TEMPERATURE", 2},
  {"TPC_PT_447_TEMPERATURE", 3},
  {"TPC_PT_477_TEMPERATURE", 4},
  {"TPC_PT_488_TEMPERATURE", 5},
  {"TPC_PT_537_TEMPERATURE", 6},
  {"TPC_PT_575_TEMPERATURE", 7},
  {"TPC_PT_589_TEMPERATURE", 8},
  {"TPC_PT_629_TEMPERATURE", 9},
  {"TPC_PT_664_TEMPERATURE", 10},
  {"TPC_PT_695_TEMPERATURE", 11},
  {"TPC_PT_735_TEMPERATURE", 12},
  {"TPC_PT_757_TEMPERATURE", 13},
  {"TPC_PT_797_TEMPERATURE", 14},
  {"TPC_PT_831_TEMPERATURE", 15},
  {"TPC_PT_851_TEMPERATURE", 16},
  {"TPC_PT_895_TEMPERATURE", 17},
};

Temperature::Temperature() noexcept : raw(SensorsPerSide * SIDES)
{
  for (size_t i = 0; i < raw.size(); ++i) {
    raw[i].sensorNumber = i;
  }
}

//==============================================================================
//
//
HV::HV() noexcept : voltages(2 * GEMSTACKSPERSECTOR * GEMSPERSTACK * SECTORSPERSIDE * SIDES),
                    currents(2 * GEMSTACKSPERSECTOR * GEMSPERSTACK * SECTORSPERSIDE * SIDES),
                    states(GEMSTACKSPERSECTOR * SECTORSPERSIDE * SIDES)
{
  for (size_t i = 0; i < voltages.size(); ++i) {
    voltages[i].sensorNumber = i;
    currents[i].sensorNumber = i;
  }
  for (size_t i = 0; i < states.size(); ++i) {
    states[i].sensorNumber = i;
  }
}

const std::unordered_map<HV::StackState, std::string> HV::StackStateNameMap =
  {
    {StackState::OFF, "OFF"},
    {StackState::STBY_CONFIGURED, "STBY_CONFIGURED"},
    {StackState::INTERMEDIATE, "INTERMEDIATE"},
    {StackState::ON, "ON"},
    {StackState::ERROR, "ERROR"},
    {StackState::ERROR_LOCAL, "ERROR_LOCAL"},
    {StackState::SOFT_INTERLOCK, "SOFT_INTERLOCK"},
    {StackState::INTERLOCK, "INTERLOCK"},
    {StackState::RAMPIG_UP_LOW, "RAMPIG_UP_LOW"},
    {StackState::RAMPIG_DOWN_LOW, "RAMPIG_DOWN_LOW"},
    {StackState::RAMPIG_UP, "RAMPIG_UP"},
    {StackState::RAMPIG_DOWN, "RAMPIG_DOWN"},
    {StackState::MIXED, "MIXED"},
    {StackState::NO_CONTROL, "NO_CONTROL"},
};

TimeStampType Gas::getMinTime() const
{
  constexpr auto max = std::numeric_limits<dcs::TimeStampType>::max();
  const std::vector<TimeStampType> times{
    neon.data.size() ? neon.data.front().time : max,
    co2.data.size() ? co2.data.front().time : max,
    n2.data.size() ? n2.data.front().time : max,
    argon.data.size() ? argon.data.front().time : max,
    h2o.data.size() ? h2o.data.front().time : max,
    o2.data.size() ? o2.data.front().time : max,
    h2oSensor.data.size() ? h2oSensor.data.front().time : max,
    o2Sensor.data.size() ? o2Sensor.data.front().time : max,
  };

  return *std::min_element(times.begin(), times.end());
}

TimeStampType Gas::getMaxTime() const
{
  constexpr auto min = 0;
  const std::vector<TimeStampType> times{
    neon.data.size() ? neon.data.back().time : min,
    co2.data.size() ? co2.data.back().time : min,
    n2.data.size() ? n2.data.back().time : min,
    argon.data.size() ? argon.data.back().time : min,
    h2o.data.size() ? h2o.data.back().time : min,
    o2.data.size() ? o2.data.back().time : min,
    h2oSensor.data.size() ? h2oSensor.data.back().time : min,
    o2Sensor.data.size() ? o2Sensor.data.back().time : min,
  };

  return *std::max_element(times.begin(), times.end());
}

TimeStampType Pressure::getMinTime() const
{
  constexpr auto max = std::numeric_limits<dcs::TimeStampType>::max();
  const std::vector<TimeStampType> times{
    cavernAtmosPressure.data.size() ? cavernAtmosPressure.data.front().time : max,
    cavernAtmosPressure2.data.size() ? cavernAtmosPressure2.data.front().time : max,
    surfaceAtmosPressure.data.size() ? surfaceAtmosPressure.data.front().time : max,
  };

  return *std::min_element(times.begin(), times.end());
}

TimeStampType Pressure::getMaxTime() const
{
  constexpr auto min = 0;
  const std::vector<TimeStampType> times{
    cavernAtmosPressure.data.size() ? cavernAtmosPressure.data.back().time : min,
    cavernAtmosPressure2.data.size() ? cavernAtmosPressure2.data.back().time : min,
    surfaceAtmosPressure.data.size() ? surfaceAtmosPressure.data.back().time : min,
  };

  return *std::max_element(times.begin(), times.end());
}

bool Temperature::makeFit(TLinearFitter& fitter, const int nDim, std::vector<double>& xVals, std::vector<double>& temperatures)
{
  const int minPointsForFit = 5;
  if (temperatures.empty() || (temperatures.size() < minPointsForFit)) {
    LOGP(warning, "Number of points {} for fit smaller than minimum of {}!", temperatures.size(), minPointsForFit);
    return false;
  }

  fitter.ClearPoints();
  fitter.AssignData(temperatures.size(), nDim, xVals.data(), temperatures.data());
  int status = fitter.Eval();
  if (status == 1) {
    LOGP(warning, "Fit failed!");
    return false;
  }
  return true;
}

void Temperature::fitTemperature(Side side, dcs::TimeStampType fitInterval, const bool roundToInterval)
{
  // clear old data
  auto& stats = (side == Side::A) ? statsA : statsC;
  stats.clear();

  // temperature fits in x-y
  const int nDim = 2;
  TLinearFitter fitter(nDim, "1 ++ x0 ++ x1", "");
  std::array<size_t, dcs::Temperature::SensorsPerSide> startPos{};
  const size_t sensorOffset = (side == Side::C) ? dcs::Temperature::SensorsPerSide : 0;

  const dcs::TimeStampType refTime = getMinTime(raw, fitInterval, roundToInterval);
  const dcs::TimeStampType refTimeMax = getMaxTime(raw);

  // calculate number of intervals and see if the last interval should be merged into the previous one
  const int lastIntervalDuration = (refTimeMax - refTime) % fitInterval;

  // process the last interval only if it contains more than 50% of the interval duration
  const bool procLastInt = (lastIntervalDuration / fitInterval > 0.5);
  int numIntervals = (refTimeMax - refTime) / fitInterval + procLastInt;
  if (numIntervals == 0) {
    numIntervals = 1;
  }

  // buffer for fit values
  std::vector<double> xVals;
  std::vector<double> temperatures;
  xVals.reserve(2 * 1000);
  temperatures.reserve(1000);

  for (int interval = 0; interval < numIntervals; ++interval) {
    const dcs::TimeStampType timeStart = refTime + interval * fitInterval;

    // clear buffer
    xVals.clear();
    temperatures.clear();

    // TODO: check if we should use refTime
    dcs::TimeStampType firstTime = std::numeric_limits<dcs::TimeStampType>::max();
    dcs::TimeStampType LastTime = 0;

    for (size_t iSensor = 0; iSensor < dcs::Temperature::SensorsPerSide; ++iSensor) {
      const auto& sensor = raw[iSensor + sensorOffset];

      LOGP(debug, "sensor {}, start {}, size {}", sensor.sensorNumber, startPos[iSensor], sensor.data.size());
      while (startPos[iSensor] < sensor.data.size()) {
        const auto& dataPoint = sensor.data[startPos[iSensor]];
        if (((dataPoint.time - timeStart) >= fitInterval) && (interval != numIntervals - 1)) {
          LOGP(debug, "sensor {}, {} - {} >= {}", sensor.sensorNumber, dataPoint.time, timeStart, fitInterval);
          break;
        }
        firstTime = std::min(firstTime, dataPoint.time);
        LastTime = std::max(LastTime, dataPoint.time);
        const auto temperature = dataPoint.value;
        // sanity check
        ++startPos[iSensor];
        if (temperature < 15 || temperature > 25) {
          continue;
        }
        const auto& pos = dcs::Temperature::SensorPosition[iSensor + sensorOffset];
        xVals.emplace_back(pos.x);
        xVals.emplace_back(pos.y);
        temperatures.emplace_back(temperature);
      }
    }
    if (firstTime < std::numeric_limits<dcs::TimeStampType>::max() && !temperatures.empty()) {
      const bool fitOk = makeFit(fitter, nDim, xVals, temperatures);
      if (!fitOk) {
        continue;
      }
      auto& stat = stats.data.emplace_back();
      stat.time = (firstTime + LastTime) / 2;
      stat.value.mean = fitter.GetParameter(0);
      stat.value.gradX = fitter.GetParameter(1);
      stat.value.gradY = fitter.GetParameter(2);

      // check if data contains outliers
      const float maxDeltaT = 1;
      const float meanTemp = fitter.GetParameter(0);
      const bool isDataGood = std::all_of(temperatures.begin(), temperatures.end(), [meanTemp, maxDeltaT](double t) { return std::abs(t - meanTemp) < maxDeltaT; });

      // do second iteration only in case of outliers
      if (!isDataGood) {
        std::vector<double> xVals2;
        std::vector<double> temperatures2;
        xVals2.reserve(xVals.size());
        temperatures2.reserve(temperatures.size());
        for (int i = 0; i < temperatures.size(); ++i) {
          if (std::abs(temperatures[i] - meanTemp) < maxDeltaT) {
            const int idx = 2 * i;
            xVals2.emplace_back(xVals[idx]);
            xVals2.emplace_back(xVals[idx + 1]);
            temperatures2.emplace_back(temperatures[i]);
          }
        }
        const bool fitOk2 = makeFit(fitter, nDim, xVals2, temperatures2);
        if (fitOk2) {
          stat.value.mean = fitter.GetParameter(0);
          stat.value.gradX = fitter.GetParameter(1);
          stat.value.gradY = fitter.GetParameter(2);
        }
      }
    }
  }
}

void Pressure::fill(std::string_view sensor, const TimeStampType time, const DataType value)
{
  if (sensor == "CavernAtmosPressure") {
    cavernAtmosPressure.fill(time, value);
  } else if (sensor == "CavernAtmosPressure2") {
    cavernAtmosPressure2.fill(time, value);
  } else if (sensor == "SurfaceAtmosPressure") {
    surfaceAtmosPressure.fill(time, value);
  } else {
    LOGP(warning, "Unknown pressure sensor {}", sensor);
  }
}

void Pressure::sortAndClean(float pMin, float pMax)
{
  cavernAtmosPressure.sortAndClean();
  cavernAtmosPressure2.sortAndClean();
  surfaceAtmosPressure.sortAndClean();

  auto removeOutliers = [](auto& dataVec, auto minVal, auto maxVal) {
    dataVec.erase(
      std::remove_if(dataVec.begin(), dataVec.end(),
                     [minVal, maxVal](const auto& dp) {
                       return (dp.value < minVal || dp.value > maxVal);
                     }),
      dataVec.end());
  };

  removeOutliers(cavernAtmosPressure.data, pMin, pMax);
  removeOutliers(cavernAtmosPressure2.data, pMin, pMax);
  removeOutliers(surfaceAtmosPressure.data, pMin, pMax);
}

void Pressure::clear()
{
  cavernAtmosPressure.clear();
  cavernAtmosPressure2.clear();
  surfaceAtmosPressure.clear();
  robustPressure = RobustPressure();
}

void Pressure::append(const Pressure& other)
{
  cavernAtmosPressure.append(other.cavernAtmosPressure);
  cavernAtmosPressure2.append(other.cavernAtmosPressure2);
  surfaceAtmosPressure.append(other.surfaceAtmosPressure);
}

void fillBuffer(std::pair<std::vector<float>, std::vector<TimeStampType>>& buffer, const std::pair<std::vector<float>, std::vector<TimeStampType>>& values, TimeStampType tStart, const int minPoints)
{
  const auto itStartBuff = std::lower_bound(buffer.second.begin(), buffer.second.end(), tStart);
  size_t idxStartBuffer = std::distance(buffer.second.begin(), itStartBuff);
  if (buffer.first.size() - idxStartBuffer < minPoints) {
    if (buffer.first.size() < minPoints) {
      idxStartBuffer = 0;
    } else {
      idxStartBuffer = buffer.first.size() - minPoints;
    }
  }

  std::pair<std::vector<float>, std::vector<TimeStampType>> buffTmp;
  auto& [buffVals, buffTimes] = buffTmp;

  // Preallocate enough capacity to avoid reallocations
  buffVals.reserve(buffer.first.size() - idxStartBuffer + values.first.size());
  buffTimes.reserve(buffer.second.size() - idxStartBuffer + values.second.size());
  // Insert the kept part of the old buffer
  buffVals.insert(buffVals.end(), buffer.first.begin() + idxStartBuffer, buffer.first.end());
  buffTimes.insert(buffTimes.end(), buffer.second.begin() + idxStartBuffer, buffer.second.end());
  // Insert the new values
  buffVals.insert(buffVals.end(), values.first.begin(), values.first.end());
  buffTimes.insert(buffTimes.end(), values.second.begin(), values.second.end());

  // this should not happen
  if (!std::is_sorted(buffTimes.begin(), buffTimes.end())) {
    LOGP(info, "Pressure buffer not sorted after filling - sorting it");
    std::vector<size_t> idx(buffTimes.size());
    o2::math_utils::SortData(buffTimes, idx);
    o2::math_utils::Reorder(buffVals, idx);
    o2::math_utils::Reorder(buffTimes, idx);
  }

  buffer = std::move(buffTmp);
}

void Pressure::makeRobustPressure(TimeStampType timeInterval, TimeStampType timeIntervalRef, TimeStampType tStart, TimeStampType tEnd, const int nthreads)
{
  const auto surfaceAtmosPressurePair = surfaceAtmosPressure.getPairOfVector();
  const auto cavernAtmosPressurePair = cavernAtmosPressure.getPairOfVector();
  const auto cavernAtmosPressure2Pair = cavernAtmosPressure2.getPairOfVector();

  // round to second
  tStart = tStart / 1000 * 1000;
  const TimeStampType tStartRef = (tStart - timeIntervalRef);
  const int minPointsRef = 50;
  fillBuffer(mCavernAtmosPressure1Buff, cavernAtmosPressurePair, tStartRef, minPointsRef);
  fillBuffer(mCavernAtmosPressure2Buff, cavernAtmosPressure2Pair, tStartRef, minPointsRef);
  fillBuffer(mSurfaceAtmosPressureBuff, surfaceAtmosPressurePair, tStartRef, minPointsRef);

  int nIntervals = std::round((tEnd - tStart) / timeInterval);
  if (nIntervals == 0) {
    nIntervals = 1; // at least one interval
  }
  std::vector<ULong64_t> times;
  times.reserve(nIntervals);
  for (int i = 0; i < nIntervals; ++i) {
    times.emplace_back(tStart + (i + 0.5) * timeInterval);
  }

  /// minimum number of points in the interval - otherwise use the n closest points
  const int minPoints = 4;
  const auto cavernAtmosPressureStats = o2::math_utils::getRollingStatistics(mCavernAtmosPressure1Buff.second, mCavernAtmosPressure1Buff.first, times, timeInterval, nthreads, minPoints, minPoints);
  const auto cavernAtmosPressure2Stats = o2::math_utils::getRollingStatistics(mCavernAtmosPressure2Buff.second, mCavernAtmosPressure2Buff.first, times, timeInterval, nthreads, minPoints, minPoints);
  const auto surfaceAtmosPressureStats = o2::math_utils::getRollingStatistics(mSurfaceAtmosPressureBuff.second, mSurfaceAtmosPressureBuff.first, times, timeInterval, nthreads, minPoints, minPoints);

  // subtract the moving median values from the different sensors if they are ok
  std::pair<std::vector<float>, std::vector<TimeStampType>> cavernAtmosPressure12;
  std::pair<std::vector<float>, std::vector<TimeStampType>> cavernAtmosPressure1S;
  std::pair<std::vector<float>, std::vector<TimeStampType>> cavernAtmosPressure2S;
  cavernAtmosPressure12.first.reserve(nIntervals);
  cavernAtmosPressure1S.first.reserve(nIntervals);
  cavernAtmosPressure2S.first.reserve(nIntervals);
  cavernAtmosPressure12.second.reserve(nIntervals);
  cavernAtmosPressure1S.second.reserve(nIntervals);
  cavernAtmosPressure2S.second.reserve(nIntervals);

  for (int i = 0; i < nIntervals; i++) {
    // coarse check if data is close by
    const int maxDist = 600 * 1000;
    const bool cavernOk = (cavernAtmosPressureStats.median[i] > 0) && (cavernAtmosPressureStats.closestDistanceL[i] < maxDist) && (cavernAtmosPressureStats.closestDistanceR[i] < maxDist);
    const bool cavern2Ok = (cavernAtmosPressure2Stats.median[i] > 0) && (cavernAtmosPressure2Stats.closestDistanceL[i] < maxDist) && (cavernAtmosPressure2Stats.closestDistanceR[i] < maxDist);
    const bool surfaceOk = (surfaceAtmosPressureStats.median[i] > 0) && (surfaceAtmosPressureStats.closestDistanceL[i] < maxDist) && (surfaceAtmosPressureStats.closestDistanceR[i] < maxDist);

    if (cavernOk && cavern2Ok) {
      cavernAtmosPressure12.first.emplace_back(cavernAtmosPressureStats.median[i] - cavernAtmosPressure2Stats.median[i]);
      cavernAtmosPressure12.second.emplace_back(times[i]);
    }
    if (cavernOk && surfaceOk) {
      cavernAtmosPressure1S.first.emplace_back(cavernAtmosPressureStats.median[i] - surfaceAtmosPressureStats.median[i]);
      cavernAtmosPressure1S.second.emplace_back(times[i]);
    }
    if (cavern2Ok && surfaceOk) {
      cavernAtmosPressure2S.first.emplace_back(cavernAtmosPressure2Stats.median[i] - surfaceAtmosPressureStats.median[i]);
      cavernAtmosPressure2S.second.emplace_back(times[i]);
    }
  }

  fillBuffer(mPressure12Buff, cavernAtmosPressure12, tStartRef, minPointsRef);
  fillBuffer(mPressure1SBuff, cavernAtmosPressure1S, tStartRef, minPointsRef);
  fillBuffer(mPressure2SBuff, cavernAtmosPressure2S, tStartRef, minPointsRef);

  // get long term median of diffs - this is used for normalization of the pressure values -
  const auto cavernAtmosPressure12Stats = o2::math_utils::getRollingStatistics(mPressure12Buff.second, mPressure12Buff.first, times, timeIntervalRef, nthreads, 3, minPointsRef);
  const auto cavernAtmosPressure1SStats = o2::math_utils::getRollingStatistics(mPressure1SBuff.second, mPressure1SBuff.first, times, timeIntervalRef, nthreads, 3, minPointsRef);
  const auto cavernAtmosPressure2SStats = o2::math_utils::getRollingStatistics(mPressure2SBuff.second, mPressure2SBuff.first, times, timeIntervalRef, nthreads, 3, minPointsRef);

  // calculate diffs of median values
  const float maxDist = 20 * timeInterval;
  const float maxDiff = 0.2;
  std::pair<std::vector<float>, std::vector<TimeStampType>> robustPressureTmp;
  robustPressureTmp.first.reserve(nIntervals);
  robustPressureTmp.second.reserve(nIntervals);
  std::vector<uint8_t> isOk(nIntervals);

  for (int i = 0; i < nIntervals; ++i) {
    // difference beween pressure values corrected for the long term median
    const float delta12 = cavernAtmosPressureStats.median[i] - cavernAtmosPressure2Stats.median[i] - cavernAtmosPressure12Stats.median[i];
    const float delta1S = cavernAtmosPressureStats.median[i] - surfaceAtmosPressureStats.median[i] - cavernAtmosPressure1SStats.median[i];
    const float delta2S = cavernAtmosPressure2Stats.median[i] - surfaceAtmosPressureStats.median[i] - cavernAtmosPressure2SStats.median[i];

    const auto distCavernAtmosPressureL = cavernAtmosPressureStats.closestDistanceL[i];
    const auto distCavernAtmosPressure2L = cavernAtmosPressure2Stats.closestDistanceL[i];
    const auto distSurfaceAtmosPressureL = surfaceAtmosPressureStats.closestDistanceL[i];
    const auto distCavernAtmosPressureR = cavernAtmosPressureStats.closestDistanceR[i];
    const auto distCavernAtmosPressure2R = cavernAtmosPressure2Stats.closestDistanceR[i];
    const auto distSurfaceAtmosPressureR = surfaceAtmosPressureStats.closestDistanceR[i];

    // check if data is ok
    const bool cavernDistOk = (cavernAtmosPressureStats.median[i] > 0) && ((distCavernAtmosPressureL < maxDist) || (distCavernAtmosPressureR < maxDist));
    const bool cavern2DistOk = (cavernAtmosPressure2Stats.median[i] > 0) && ((distCavernAtmosPressure2L < maxDist) || (distCavernAtmosPressure2R < maxDist));
    const bool surfaceDistOk = (surfaceAtmosPressureStats.median[i] > 0) && ((distSurfaceAtmosPressureL < maxDist) || (distSurfaceAtmosPressureR < maxDist));
    const bool onlyOneSensor = (cavernDistOk + cavern2DistOk + surfaceDistOk) == 1; // check if only 1 sensor exists, if so use that sensor

    uint8_t maskIsOkTmp = 0;
    const int cavernBit = 0;  // val 1
    const int cavern2Bit = 1; // val 2
    const int surfaceBit = 2; // val 4

    // check if ratio sensor 1 and 2 are good
    // maskIsOkTmp = 3
    if (((std::abs(delta12) < maxDiff) && (cavernDistOk && cavern2DistOk)) || onlyOneSensor) {
      if (cavernDistOk) {
        maskIsOkTmp |= (1 << cavernBit);
      }
      if (cavern2DistOk) {
        maskIsOkTmp |= (1 << cavern2Bit);
      }
    }

    // check if ratio sensor 1 and surface are good
    // maskIsOkTmp = 5
    if ((std::abs(delta1S) < maxDiff) && ((cavernDistOk && surfaceDistOk)) || onlyOneSensor) {
      if (cavernDistOk) {
        maskIsOkTmp |= (1 << cavernBit);
      }
      if (surfaceDistOk) {
        maskIsOkTmp |= (1 << surfaceBit);
      }
    }

    // check if ratio sensor 2 and surface are good
    // maskIsOkTmp = 6
    if ((std::abs(delta2S) < maxDiff) && ((cavern2DistOk && surfaceDistOk)) || onlyOneSensor) {
      if (cavern2DistOk) {
        maskIsOkTmp |= (1 << cavern2Bit);
      }
      if (surfaceDistOk) {
        maskIsOkTmp |= (1 << surfaceBit);
      }
    }

    // calculate robust pressure
    float pressure = 0;
    int pressureCount = 0;
    if ((maskIsOkTmp >> cavernBit) & 1) {
      pressure += cavernAtmosPressureStats.median[i];
      pressureCount++;
    }

    if ((maskIsOkTmp >> cavern2Bit) & 1) {
      pressure += cavernAtmosPressure2Stats.median[i] + cavernAtmosPressure12Stats.median[i];
      pressureCount++;
    }

    if ((maskIsOkTmp >> surfaceBit) & 1) {
      pressure += surfaceAtmosPressureStats.median[i] + cavernAtmosPressure1SStats.median[i];
      pressureCount++;
    }

    isOk[i] = maskIsOkTmp;
    if (pressureCount > 0) {
      pressure /= pressureCount;
      robustPressureTmp.first.emplace_back(pressure);
      robustPressureTmp.second.emplace_back(times[i]);
    }
  }

  fillBuffer(mRobPressureBuff, robustPressureTmp, tStartRef, minPointsRef);

  RobustPressure& pOut = robustPressure;
  pOut.surfaceAtmosPressure = std::move(surfaceAtmosPressureStats);
  pOut.cavernAtmosPressure2 = std::move(cavernAtmosPressure2Stats);
  pOut.cavernAtmosPressure = std::move(cavernAtmosPressureStats);
  pOut.cavernAtmosPressure12 = std::move(cavernAtmosPressure12Stats);
  pOut.cavernAtmosPressure1S = std::move(cavernAtmosPressure1SStats);
  pOut.cavernAtmosPressure2S = std::move(cavernAtmosPressure2SStats);
  pOut.isOk = std::move(isOk);
  pOut.robustPressure = o2::math_utils::getRollingStatistics(mRobPressureBuff.second, mRobPressureBuff.first, times, timeInterval, nthreads, 1, 5).median;
  pOut.time = std::move(times);
  pOut.timeInterval = timeInterval;
  pOut.timeIntervalRef = timeIntervalRef;
  pOut.maxDist = maxDist;
  pOut.maxDiff = maxDiff;
}

void Pressure::setAliases(TTree* tree)
{
  tree->SetAlias("cavernDistOk", "robustPressure.cavernAtmosPressure.median>0 && (robustPressure.cavernAtmosPressure.closestDistanceR<robustPressure.maxDist || robustPressure.cavernAtmosPressure.closestDistanceL<robustPressure.maxDist)");
  tree->SetAlias("cavern2DistOk", "robustPressure.cavernAtmosPressure2.median>0 && (robustPressure.cavernAtmosPressure2.closestDistanceR<robustPressure.maxDist || robustPressure.cavernAtmosPressure2.closestDistanceL<robustPressure.maxDist)");
  tree->SetAlias("surfaceDistOk", "robustPressure.surfaceAtmosPressure.median>0 && (robustPressure.surfaceAtmosPressure.closestDistanceR<robustPressure.maxDist || robustPressure.surfaceAtmosPressure.closestDistanceL<robustPressure.maxDist)");
  tree->SetAlias("onlyOneSensor", "(cavernDistOk + cavern2DistOk + surfaceDistOk) == 1");
  tree->SetAlias("delta12", "robustPressure.cavernAtmosPressure.median - robustPressure.cavernAtmosPressure2.median - robustPressure.cavernAtmosPressure12.median");
  tree->SetAlias("delta1S", "robustPressure.cavernAtmosPressure.median - robustPressure.surfaceAtmosPressure.median - robustPressure.cavernAtmosPressure1S.median");
  tree->SetAlias("delta2S", "robustPressure.surfaceAtmosPressure.median - robustPressure.cavernAtmosPressure2.median - robustPressure.cavernAtmosPressure2S.median");
  tree->SetAlias("delta12_Ok", "abs(delta12)<robustPressure.maxDiff");
  tree->SetAlias("delta1S_Ok", "abs(delta1S)<robustPressure.maxDiff");
  tree->SetAlias("delta2S_Ok", "abs(delta2S)<robustPressure.maxDiff");
}
