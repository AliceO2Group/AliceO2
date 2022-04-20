
#ifndef HMPIDDATAFORMATS_H
#define HMPIDDATAFORMATS_H

#include <algorithm>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>
#include <array>
#include <unordered_map>
#include <cstdlib>

#include "Rtypes.h"

#include "Framework/Logger.h"
//#include "DataFormatsTPC/Defs.h"
//using namespace o2::hmpid; ?? 


namespace o2::hmpid::dcs
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

  void fill(const TimeStampType time, const T& value) { data.emplace_back(DPType{time, value}); }


  void fill(const DPType& dataPoint) { data.emplace_back(dataPoint); }
  void fillTemp(std::size_t chamberNumber,std::size_t radiatorNumber,const TimeStampType time, const T& value);
  void fillHV(std::size_t chamberNumber,std::size_t sectorNumber,const TimeStampType  time, const T& value);
  void fillPressure(std::size_t chamberNumber,const TimeStampType  time, const T& value);
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

  /// return value at the last valid time stamp
  ///
  /// values are valid unitl the next time stamp
  const T& getValueForTime(const TimeStampType timeStamp) const
  {
    const auto i = std::upper_bound(data.begin(), data.end(), DPType{timeStamp, {}});
    return (i == data.begin()) ? (*i).value : (*(i - 1)).value;
  };

  ClassDefNV(DataPointVector, 1);
};

template <typename T>
void doSortAndClean(std::vector<DataPointVector<T>>& dataVector)
{
  for (auto& data : dataVector) {
    data.sortAndClean();
  }
}

template <typename T>
void doClear(std::vector<DataPointVector<T>>& dataVector)
{
  for (auto& data : dataVector) {
    data.clear();
  }
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

 /*
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

  static constexpr auto& getSensorPosition(const size_t sensor) { return SensorPosition[sensor]; } */

  struct Stats {
    DataType mean{};  ///< average temperature in K
    DataType gradX{}; ///< horizontal temperature gradient in K/cm
    DataType gradY{}; ///< vertical temperature gradient in K/cm

    ClassDefNV(Stats, 1);
  };
  using StatsDPs = DataPointVector<Stats>;

  StatsDPs statsA;          ///< statistics fit values per integration time A-Side
  StatsDPs statsC;          ///< statistics fit values per integration time C-Side
  std::vector<RawDPsF> raw; ///< raw temperature values from DCS for
  /*
  const Stats& getStats(const Side s, const TimeStampType timeStamp) const
  {
    return (s == Side::A) ? statsA.getValueForTime(timeStamp) : statsC.getValueForTime(timeStamp);
  } 
  */ 
  //aliasStr,chamberNumber,radiatorNumber, time, value
  void fill(std::string sensor,std::size_t chamberNumber,std::size_t radiatorNumber, const TimeStampType time, const DataType temperature)
  {
    raw[SensorNameMap.at(sensor.data())].fillTemp(chamberNumber,radiatorNumber,time, temperature);
  };

  void sortAndClean()
  {
    doSortAndClean(raw);
  }

  void clear()
  {
    doClear(raw);
  }

  ClassDefNV(Temperature, 1);
}; // end struct temperature


struct HV {

  HV()
  noexcept;
  
  std::vector<int> t[7][3]; 
  
  //static const std::unordered_map<StackState, std::string> StackStateNameMap; //!< map state to string
  
  //using RawDPsState = DataPointVector<StackState>;

  //(aliasStr,chamberNumber,sectorNumber, time, value);
  void fill(std::string sensor,std::size_t chamberNumber,std::size_t sectorNumber,    const TimeStampType time, const DataType HV)
  {
    t[chamberNumber][sectorNumber].push_back(5);
    // TODO: check value for validity
    //states[stackID.getIndex()].fill(time, static_cast<StackState>(value));
  }
  /*
  void sortAndClean()
  {
    doSortAndClean(states);
  }

  void clear()
  {
    doClear(states);
  } */

  ClassDefNV(HV, 1);
}; // end struct HV

struct Pressure {

  Pressure()
  noexcept;
  
  std::vector<int> t; 
  
  //static const std::unordered_map<StackState, std::string> StackStateNameMap; //!< map state to string
  
  //using RawDPsState = DataPointVector<StackState>;

  //(aliasStr,chamberNumber,sectorNumber, time, value);
  void fill(std::string sensor, const TimeStampType time, const DataType HV)
  {
    t.push_back(5);
    // TODO: check value for validity
    //states[stackID.getIndex()].fill(time, static_cast<StackState>(value));
  }
  /*
  void sortAndClean()
  {
    doSortAndClean(states);
  }

  void clear()
  {
    doClear(states);
  } */

  ClassDefNV(Pressure, 1);
}; // end struct HV

struct ChamberPressure {

  ChamberPressure()
  noexcept;
  
  std::vector<int> t[7]; 
  
  //static const std::unordered_map<StackState, std::string> StackStateNameMap; //!< map state to string
  
  //using RawDPsState = DataPointVector<StackState>;

  //(aliasStr,chamberNumber,sectorNumber, time, value);
  void fill(std::string sensor,std::size_t chamberNumber,    const TimeStampType time, const DataType HV)
  {
    t[chamberNumber].push_back(5);
    // TODO: check value for validity
    //states[stackID.getIndex()].fill(time, static_cast<StackState>(value));
  }
  /*
  void sortAndClean()
  {
    doSortAndClean(states);
  }

  void clear()
  {
    doClear(states);
  } */

  ClassDefNV(ChamberPressure, 1);
}; // end struct HV

} // end namespace 

#endif
