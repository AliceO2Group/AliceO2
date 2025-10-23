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

/// @file   DCS.h
/// @author Jens Wiechula
/// @brief  DCS data point data formats

#ifndef TPC_DCSCalibData_H_
#define TPC_DCSCalibData_H_

#include <algorithm>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>
#include <numeric>
#include <array>
#include <unordered_map>
#include <cstdlib>

#include "Rtypes.h"

#include "Framework/Logger.h"
#include "DataFormatsTPC/Defs.h"
#include "MathUtils/fit.h"

using namespace o2::tpc;

class TLinearFitter;
class TTree;

namespace o2::tpc::dcs
{

using DataType = float;

using TimeStampType = uint64_t;

/// Data point keeping value and time information
///
///
template <typename T>
struct DataPoint {
  TimeStampType time;
  T value;

  bool equalTime(const DataPoint& other) const { return time == other.time; }
  bool operator<(const DataPoint& other) const { return time < other.time; }
  bool operator<(const TimeStampType timeStamp) const { return time < timeStamp; }
  DataPoint operator+(const DataPoint& other) const { return DataPoint{(time + other.time) / TimeStampType{2}, value + other.value}; }
  DataPoint operator/(const DataType denom) const { return DataPoint{time, value / denom}; }

  ClassDefNV(DataPoint, 1);
};

/// Vector of data points at different time stamps
///
///
template <typename T>
struct DataPointVector {
  using DPType = DataPoint<T>;
  uint32_t sensorNumber{};
  std::vector<DPType> data;

  /// \brief convert data points to a vector of pairs: pair.first -> data and pair.second -> time
  auto getPairOfVector() const
  {
    std::pair<std::vector<T>, std::vector<TimeStampType>> pairs;
    pairs.first.reserve(data.size());
    pairs.second.reserve(data.size());
    for (const auto& dp : data) {
      pairs.first.emplace_back(dp.value);
      pairs.second.emplace_back(dp.time);
    }
    return pairs;
  }

  void fill(const TimeStampType time, const T& value) { data.emplace_back(DPType{time, value}); }

  void fill(const DPType& dataPoint) { data.emplace_back(dataPoint); }

  void sort() { std::sort(data.begin(), data.end()); }

  void sortAndClean()
  {
    sort();
    data.erase(
      std::unique(data.begin(), data.end(),
                  [](const auto& dp1, const auto& dp2) {
                    return dp1.time == dp2.time;
                  }),
      data.end());
    data.shrink_to_fit();
  }

  void clear() { data.clear(); }

  void append(const DataPointVector<T>& other)
  {
    data.insert(data.end(), other.data.begin(), other.data.end());
  }

  /// return value at the last valid time stamp
  ///
  /// values are valid unitl the next time stamp
  const T& getValueForTime(const TimeStampType timeStamp) const
  {
    const auto i = std::upper_bound(data.begin(), data.end(), DPType{timeStamp, {}});
    return (i == data.begin()) ? (*i).value : (*(i - 1)).value;
  }

  /// calculate average value for `timeStamp`, extending the range by +- range elements
  const T getAverageValueForTime(const TimeStampType timeStamp, const long range) const
  {
    return getAverageValueForTime(timeStamp, timeStamp, range);
  }

  /// calculate average value between `from` and `unil`, extending the range by +- `range` elements
  const std::pair<T, long> getSumAndPoints(const TimeStampType from, const TimeStampType until, const long range) const
  {
    const auto iFrom = std::upper_bound(data.begin(), data.end(), DPType{from, {}});
    const auto iUntil = (from == until) ? iFrom : std::upper_bound(data.begin(), data.end(), DPType{until, {}});
    const auto distFrom = std::distance(data.begin(), iFrom);
    const auto distUntil = std::distance(iUntil, data.end());
    const auto nFrom = std::min(distFrom, range + 1);
    const auto nUntil = std::min(distUntil, range);
    const auto nPoints = std::distance(iFrom - nFrom, iUntil + nUntil);
    return {std::accumulate(iFrom - nFrom, iUntil + nUntil, DPType{(*(iFrom - nFrom)).time, T{}}).value, nPoints};
  }

  const T getAverageValueForTime(const TimeStampType from, const TimeStampType until, const long range) const
  {
    const auto sumAndPoints = getSumAndPoints(from, until, range);
    return sumAndPoints.first / static_cast<float>(sumAndPoints.second);
  }

  ClassDefNV(DataPointVector, 1);
};

template <typename T>
void doSortAndClean(std::vector<dcs::DataPointVector<T>>& dataVector)
{
  for (auto& data : dataVector) {
    data.sortAndClean();
  }
}

template <typename T>
void doClear(std::vector<dcs::DataPointVector<T>>& dataVector)
{
  for (auto& data : dataVector) {
    data.clear();
  }
}

template <typename T>
void doAppend(std::vector<dcs::DataPointVector<T>>& a, const std::vector<dcs::DataPointVector<T>>& b)
{
  if (a.size() != b.size()) {
    LOGP(warning, "Trying to append std::vector<dcs::DataPointVector<T>>s of different size: {} != {}", a.size(), b.size());
  }
  for (size_t i = 0; i < a.size(); ++i) {
    a[i].append(b[i]);
  }
}

template <typename T>
const T getAverageValueForTime(const std::vector<dcs::DataPointVector<T>>& dpVec, const TimeStampType from, const TimeStampType until, const long range)
{
  T ret{};
  long nPoints{};

  for (const auto& dps : dpVec) {
    const auto sumAndPoints = dps.getSumAndPoints(from, until, range);
    ret += sumAndPoints.first;
    nPoints += sumAndPoints.second;
  }
  return (nPoints > 0) ? ret / static_cast<float>(nPoints) : T{};
}

template <typename T>
dcs::TimeStampType getMinTime(const std::vector<dcs::DataPointVector<T>>& data, const bool roundToInterval, dcs::TimeStampType fitInterval)
{
  constexpr auto max = std::numeric_limits<dcs::TimeStampType>::max();
  dcs::TimeStampType firstTime = std::numeric_limits<dcs::TimeStampType>::max();
  for (const auto& sensor : data) {
    const auto time = sensor.data.size() ? sensor.data.front().time : max;
    firstTime = std::min(firstTime, time);
  }

  // mFitInterval is is seconds. Round to full amount.
  // if e.g. mFitInterval = 5min, then round 10:07:20.510 to 10:05:00.000
  if (roundToInterval) {
    firstTime -= (firstTime % fitInterval);
  }

  return firstTime;
}

template <typename T>
dcs::TimeStampType getMaxTime(const std::vector<dcs::DataPointVector<T>>& data)
{
  constexpr auto min = 0;
  dcs::TimeStampType lastTime = 0;
  for (const auto& sensor : data) {
    const auto time = sensor.data.size() ? sensor.data.back().time : 0;
    lastTime = std::max(lastTime, time);
  }

  // mFitInterval is is seconds. Round to full amount.
  // if e.g. mFitInterval = 5min, then round 10:07:20.510 to 10:05:00.000
  // TODO: fix this
  // if (mRoundToInterval) {
  // lastTime -= (lastTime % mFitInterval);
  //}

  return lastTime;
}

using RawDPsF = DataPointVector<float>;
// using RawDPsI = DataPointVector<int>;

/// Temerature value store
///
///
struct Temperature {
  struct Position {
    float x;
    float y;
  };

  Temperature() noexcept;

  static constexpr int SensorsPerSide = 9; ///< number of temperature sensors in the active volume per side

  static const std::unordered_map<std::string, int> SensorNameMap;

  static constexpr std::array<Position, SensorsPerSide * SIDES> SensorPosition{{
    {211.40f, 141.25f},
    {82.70f, 227.22f},
    {-102.40f, 232.72f},
    {-228.03f, 112.45f},
    {-246.96f, -60.43f},
    {-150.34f, -205.04f},
    {-16.63f, -253.71f},
    {175.82f, -183.66f},
    {252.74f, -27.68f},
    {228.03f, 112.45f},
    {102.40f, 232.72f},
    {-71.15f, 244.09f},
    {-211.40f, 141.25f},
    {-252.74f, -27.68f},
    {-175.82f, -183.66f},
    {-16.63f, -253.71f},
    {150.34f, -205.04f},
    {252.74f, -27.68f},
  }};

  static constexpr auto& getSensorPosition(const size_t sensor) { return SensorPosition[sensor]; }

  /// \brief make fit of the mean temperature and gradients in time intervals
  /// \param Side TPC side for which to make the fit
  /// \param fitInterval time interval for the fits
  /// \param roundToInterval round min time
  void fitTemperature(Side side, dcs::TimeStampType fitInterval = 5 * 60 * 1000, const bool roundToInterval = false);

  struct Stats {
    DataType mean{};  ///< average temperature in K
    DataType gradX{}; ///< horizontal temperature gradient in K/cm
    DataType gradY{}; ///< vertical temperature gradient in K/cm

    Stats operator+(const Stats& other) const { return Stats{mean + other.mean, gradX + other.gradX, gradY + other.gradY}; }
    Stats operator/(const DataType val) const { return Stats{mean / val, gradX / val, gradY / val}; }

    ClassDefNV(Stats, 1);
  };
  using StatsDPs = DataPointVector<Stats>;

  StatsDPs statsA;          ///< statistics fit values per integration time A-Side
  StatsDPs statsC;          ///< statistics fit values per integration time C-Side
  std::vector<RawDPsF> raw; ///< raw temperature values from DCS for

  const Stats& getStats(const Side s, const TimeStampType timeStamp) const
  {
    return (s == Side::A) ? statsA.getValueForTime(timeStamp) : statsC.getValueForTime(timeStamp);
  }

  DataType getMeanTempRaw()
  {
    return getAverageValueForTime(raw, 0, 9999999999999, 0);
  }

  void fill(std::string_view sensor, const TimeStampType time, const DataType temperature)
  {
    raw[SensorNameMap.at(sensor.data())].fill(time, temperature);
  };

  void sortAndClean()
  {
    statsA.sortAndClean();
    statsC.sortAndClean();
    doSortAndClean(raw);
  }

  void clear()
  {
    doClear(raw);
    statsA.clear();
    statsC.clear();
  }

  void append(const Temperature& other)
  {
    statsA.append(other.statsA);
    statsC.append(other.statsC);
    doAppend(raw, other.raw);
  }

 private:
  bool makeFit(TLinearFitter& fitter, const int nDim, std::vector<double>& xVals, std::vector<double>& temperatures);

  ClassDefNV(Temperature, 1);
};

/// HV value store
///
///
struct HV {

  HV() noexcept;

  // Exmple strings
  // TPC_HV_A03_I_G1B_I
  // TPC_HV_A03_O1_G1B_I
  static constexpr size_t SidePos = 7;       ///< Position of the side identifier
  static constexpr size_t SectorPos = 8;     ///< Position of the sector number
  static constexpr size_t ROCPos = 11;       ///< Position of the readout chamber type
  static constexpr size_t GEMPos = 14;       ///< GEM position. OROC is +1
  static constexpr size_t ElectrodePos = 15; ///< Electrode type (T, B). OROC is +1

  enum class StackState : char {
    NO_CONTROL = 2,
    STBY_CONFIGURED = 3,
    OFF = 4,
    RAMPIG_DOWN = 7,
    RAMPIG_UP = 8,
    RAMPIG_DOWN_LOW = 9,
    RAMPIG_UP_LOW = 10,
    ON = 11,
    ERROR = 13,
    INTERMEDIATE = 14,
    MIXED = 19,
    INTERLOCK = 24,
    ERROR_LOCAL = 25,
    SOFT_INTERLOCK = 29,
  };

  static const std::unordered_map<StackState, std::string> StackStateNameMap; //!< map state to string

  using RawDPsState = DataPointVector<StackState>;

  std::vector<RawDPsF> voltages;   ///< voltages per GEM stack, counting is IROCs GEM1 top, bottom, GEM2 top, bottom, .. O1 ..., O2
  std::vector<RawDPsF> currents;   ///< currents per GEM stack, counting is IROCs GEM1 top, bottom, GEM2 top, bottom, .. O1 ..., O2
  std::vector<RawDPsState> states; ///< HV state per sector

  static int getSector(std::string_view sensor)
  {
    const auto sideOffset = (sensor[SidePos] == 'A') ? 0 : SECTORSPERSIDE;
    const auto sector = std::atoi(sensor.substr(SectorPos, 2).data());
    return sector + sideOffset;
  }

  static GEMstack getStack(std::string_view sensor)
  {
    if (sensor[ROCPos] == 'I') {
      return GEMstack::IROCgem;
    }
    const auto orocType = int(sensor[ROCPos + 1] - '0');
    return static_cast<GEMstack>(orocType);
  }

  /// Fill voltage and current information
  void fillUI(std::string_view sensor, const TimeStampType time, const DataType value)
  {
    const int sector = getSector(sensor);
    const auto stack = getStack(sensor);
    const auto rocOffset = int(stack != GEMstack::IROCgem);
    const auto gem = int(sensor[GEMPos + rocOffset] - '0');
    const bool isTop = sensor[ElectrodePos + rocOffset] == 'T';
    // the counting is GEM1 top, bottom, GEM2 top, bottom, ...
    const int electrode = 2 * (gem - 1) + !isTop;
    const StackID stackID{sector, stack};
    const int index = stackID.getIndex() * 2 * GEMSPERSTACK + electrode;

    const auto type = sensor.back();
    // LOGP(info, "Fill type: {}, index: {} (sec: {}, stack: {}, gem: {}, elec: {}), time: {}, value: {}", type, index, sector, stack, gem, electrode, time, value);
    if (type == 'I') {
      currents[index].fill(time, value);
    } else if (type == 'U') {
      voltages[index].fill(time, value);
    }
  }

  /// Fill stack status information
  void fillStatus(std::string_view sensor, const TimeStampType time, const uint32_t value)
  {
    const int sector = getSector(sensor);
    const auto stack = getStack(sensor);
    const StackID stackID{sector, stack};

    // TODO: check value for validity
    states[stackID.getIndex()].fill(time, static_cast<StackState>(value));
  }

  void sortAndClean()
  {
    doSortAndClean(voltages);
    doSortAndClean(currents);
    doSortAndClean(states);
  }

  void clear()
  {
    doClear(voltages);
    doClear(currents);
    doClear(states);
  }

  void append(const HV& other)
  {
    doAppend(voltages, other.voltages);
    doAppend(currents, other.currents);
    doAppend(states, other.states);
  }

  ClassDefNV(HV, 1);
};

/// Gas value store
///
///
struct Gas {
  static constexpr size_t SensorPos = 4;  ///< Position of the sensor type identifier
  static constexpr size_t TypePosGC = 7;  ///< Position of the sensor type identifier
  static constexpr size_t TypePosAn = 15; ///< Position of the sensor type identifier

  RawDPsF neon{};      ///< neon measurement from gas chromatograph
  RawDPsF co2{};       ///< CO2 measurement from gas chromatograph
  RawDPsF n2{};        ///< neon measurement from gas chromatograph
  RawDPsF argon{};     ///< argon measurement from gas chromatograph
  RawDPsF h2o{};       ///< H2O measurement from gas chromatograph
  RawDPsF o2{};        ///< O2 measurement from gas chromatograph
  RawDPsF h2oSensor{}; ///< O2 measurement from dedicated gas sensor
  RawDPsF o2Sensor{};  ///< O2 measurement from dedicated gas sensor

  void fill(std::string_view sensor, const TimeStampType time, const DataType value)
  {
    if (sensor[SensorPos] == 'G') { // check if from GC
      switch (sensor[TypePosGC]) {
        case 'N': {
          if (sensor[TypePosGC + 1] == 'E') {
            neon.fill(time, value);
          } else {
            n2.fill(time, value);
          }
          break;
        }
        case 'A':
          argon.fill(time, value);
          break;
        case 'C':
          co2.fill(time, value);
          break;
        case 'O':
          o2.fill(time, value);
          break;
        case 'W':
          h2o.fill(time, value);
          break;
        default:
          LOGP(warning, "Unknown gas sensor {}", sensor);
          break;
      }
    } else { // otherwise dedicated sensor
      switch (sensor[TypePosAn]) {
        case 'H':
          h2oSensor.fill(time, value);
          break;
        case 'O':
          o2Sensor.fill(time, value);
          break;
        default:
          LOGP(warning, "Unknown gas sensor {}", sensor);
          break;
      }
    }
  };

  void sortAndClean()
  {
    neon.sortAndClean();
    co2.sortAndClean();
    n2.sortAndClean();
    argon.sortAndClean();
    h2o.sortAndClean();
    o2.sortAndClean();
    h2oSensor.sortAndClean();
    o2Sensor.sortAndClean();
  }

  void clear()
  {
    neon.clear();
    co2.clear();
    n2.clear();
    argon.clear();
    h2o.clear();
    o2.clear();
    h2oSensor.clear();
    o2Sensor.clear();
  }

  void append(const Gas& other)
  {
    neon.append(other.neon);
    co2.append(other.co2);
    n2.append(other.n2);
    argon.append(other.argon);
    h2o.append(other.h2o);
    o2.append(other.o2);
    h2oSensor.append(other.h2oSensor);
    o2Sensor.append(other.o2Sensor);
  }

  TimeStampType getMinTime() const;

  TimeStampType getMaxTime() const;

  ClassDefNV(Gas, 1);
};

struct RobustPressure {
  using Stats = o2::math_utils::RollingStats;
  Stats surfaceAtmosPressure;        ///< rolling statistics of surface sensor
  Stats cavernAtmosPressure;         ///< rolling statistics of cavern sensor 1
  Stats cavernAtmosPressure2;        ///< rolling statistics of cavern sensor 2
  Stats cavernAtmosPressure12;       ///< rolling statistics of cavernAtmosPressure/cavernAtmosPressure2
  Stats cavernAtmosPressure1S;       ///< rolling statistics of cavernAtmosPressure/surfaceAtmosPressure
  Stats cavernAtmosPressure2S;       ///< rolling statistics of cavernAtmosPressure2/surfaceAtmosPressure
  std::vector<uint8_t> isOk;         ///< bit mask of valid sensors: cavernBit 0, cavern2Bit = 1, surfaceBit = 2
  std::vector<float> robustPressure; ///< combined robust pressure value that should be used
  std::vector<ULong64_t> time;       ///< time stamps of all pressure values
  TimeStampType timeInterval;        ///< time interval used for rolling statistics
  TimeStampType timeIntervalRef;     ///< reference time interval used for normalization of pressure sensors
  float maxDist{};                   ///< maximum allowed time distance between sensors to be accepted for robust pressure calculation
  float maxDiff{0.2f};               ///< maximum allowed pressure difference between sensors to be accepted for robust pressure calculation

  ClassDefNV(RobustPressure, 2);
};

struct Pressure {

  /// \brief fill pressure data
  /// \param sensor name of the sensor from DCS data stream
  /// \param time measurement time
  /// \param value pressure value
  void fill(std::string_view sensor, const TimeStampType time, const DataType value);

  /// sort pressure values and remove obvious outliers
  /// \param pMin min accepted pressure
  /// \param pMax max accepted pressure
  void sortAndClean(float pMin = 800, float pMax = 1100);

  /// \clear all stored data except the buffer
  void clear();

  /// append other pressure values
  void append(const Pressure& other);

  /// \return get minimum time of stored data
  TimeStampType getMinTime() const;

  /// \return get maximum time of stored data
  TimeStampType getMaxTime() const;

  /// \brief average pressure values for given time interval
  /// \param timeInterval time interval for which the pressure values are averaged
  /// \param timeIntervalRef time interval used to calculate the normalization values for the pressure
  /// \param tStart min time of the data
  /// \param tEnd max time of the data
  /// \param nthreads numbe rof threads used for some calculations
  void makeRobustPressure(TimeStampType timeInterval = 100 * 1000, TimeStampType timeIntervalRef = 24 * 60 * 1000, TimeStampType tStart = 1, TimeStampType tEnd = 0, const int nthreads = 1);

  /// set aliases for the cuts used in the calculation of the robust pressure
  static void setAliases(TTree* tree);

  RawDPsF cavernAtmosPressure{};   ///< raw pressure in the cavern from sensor 1
  RawDPsF cavernAtmosPressure2{};  ///< raw pressure in the cavern from sensor 2
  RawDPsF surfaceAtmosPressure{};  ///< raw pressure at the surface
  RobustPressure robustPressure{}; ///< combined robust pressure estimator from all three sensors

  std::pair<std::vector<float>, std::vector<TimeStampType>> mCavernAtmosPressure1Buff{}; ///<! buffer for the pressure cavern 1 sensor
  std::pair<std::vector<float>, std::vector<TimeStampType>> mCavernAtmosPressure2Buff{}; ///<! buffer for the pressure cavern 2 sensor
  std::pair<std::vector<float>, std::vector<TimeStampType>> mSurfaceAtmosPressureBuff{}; ///<! buffer for the pressure surface sensort

  std::pair<std::vector<float>, std::vector<TimeStampType>> mPressure12Buff{}; ///<! buffer for normalizing the pressure cavern 1 / cavern 2
  std::pair<std::vector<float>, std::vector<TimeStampType>> mPressure1SBuff{}; ///<! buffer for normalizing the pressure cavern 1 / surface
  std::pair<std::vector<float>, std::vector<TimeStampType>> mPressure2SBuff{}; ///<! buffer for normalizing the pressure cavern 2 / surface

  std::pair<std::vector<float>, std::vector<TimeStampType>> mRobPressureBuff{}; ///<! buffer for the robust pressure
  ClassDefNV(Pressure, 1);
};

} // namespace o2::tpc::dcs
#endif
