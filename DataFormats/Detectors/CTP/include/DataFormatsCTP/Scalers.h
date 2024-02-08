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

/// \file Scalers.h
/// \brief definition of CTPScalerRaw, CTPScalerO2
/// \author Roman Lietava

#ifndef _CTP_SCALERS_H_
#define _CTP_SCALERS_H_
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsCTP/Digits.h"
#include <map>
#include <bitset>
#include <ctime>

namespace o2
{
namespace ctp
{
/// raw scalers produced by CTP and send to O2 either via
/// - ZeroMQ published at CTP control machine
/// - CTPreadout to FLP
struct errorCounters {
  errorCounters() = default;
  void printStream(std::ostream& stream) const;
  uint32_t lmB = 0, l0B = 0, l1B = 0, lmA = 0, l0A = 0, l1A = 0;       // decreasing counters
  uint32_t lmBlmA = 0, lmAl0B = 0, l0Bl0A = 0, l0Al1B = 0, l1Bl1A = 0; // between levels countres
  uint32_t lmBlmAd1 = 0, lmAl0Bd1 = 0, l0Bl0Ad1 = 0, l0Al1Bd1 = 0, l1Bl1Ad1 = 0; // between levels countres - diff =1 - just warning
  uint32_t MAXPRINT = 3;
};
struct CTPScalerRaw {
  CTPScalerRaw() = default;
  uint32_t classIndex;
  uint32_t lmBefore;
  uint32_t lmAfter;
  uint32_t l0Before;
  uint32_t l0After;
  uint32_t l1Before;
  uint32_t l1After;
  void printStream(std::ostream& stream) const;
  ClassDefNV(CTPScalerRaw, 1);
};
/// Scalers produced from raw scalers corrected for overflow
struct CTPScalerO2 {
  CTPScalerO2() = default;
  void createCTPScalerO2FromRaw(const CTPScalerRaw& raw, const std::array<uint32_t, 6>& overfow);
  uint32_t classIndex;
  uint64_t lmBefore;
  uint64_t lmAfter;
  uint64_t l0Before;
  uint64_t l0After;
  uint64_t l1Before;
  uint64_t l1After;
  void printStream(std::ostream& stream) const;
  void printFromZero(std::ostream& stream, CTPScalerO2& scaler0) const;
  ClassDefNV(CTPScalerO2, 1);
};
struct CTPScalerRecordRaw {
  CTPScalerRecordRaw() = default;
  o2::InteractionRecord intRecord;
  double_t epochTime;
  std::vector<CTPScalerRaw> scalers;
  // std::vector<uint32_t> scalersDets;
  std::vector<uint32_t> scalersInps;
  void printStream(std::ostream& stream) const;
  ClassDefNV(CTPScalerRecordRaw, 4);
};
struct CTPScalerRecordO2 {
  CTPScalerRecordO2() = default;
  o2::InteractionRecord intRecord;
  double_t epochTime;
  std::vector<CTPScalerO2> scalers;
  // std::vector<uint64_t> scalersDets;
  std::vector<uint64_t> scalersInps;
  void printStream(std::ostream& stream) const;
  void printFromZero(std::ostream& stream, CTPScalerRecordO2& record0) const;
  ClassDefNV(CTPScalerRecordO2, 4);
};
class CTPRunScalers
{
 public:
  CTPRunScalers() = default;

  void printStream(std::ostream& stream) const;
  void printO2(std::ostream& stream) const;
  void printFromZero(std::ostream& stream) const;
  void printClasses(std::ostream& stream) const;
  std::vector<uint32_t> getClassIndexes() const;
  int getScalerIndexForClass(uint32_t cls) const;
  std::vector<CTPScalerRecordO2>& getScalerRecordO2() { return mScalerRecordO2; };
  int readScalers(const std::string& rawscalers);
  int convertRawToO2();
  int checkConsistency(const CTPScalerO2& scal0, const CTPScalerO2& scal1, errorCounters& eCnts) const;
  int checkConsistency(const CTPScalerRecordO2& rec0, const CTPScalerRecordO2& rec1, errorCounters& eCnts) const;
  void setClassMask(std::bitset<CTP_NCLASSES> classMask) { mClassMask = classMask; };
  void setDetectorMask(o2::detectors::DetID::mask_t mask) { mDetectorMask = mask; };
  void setRunNumber(uint32_t rnumber) { mRunNumber = rnumber; };
  void addScalerRacordRaw(CTPScalerRecordRaw& scalerrecordraw) { mScalerRecordRaw.push_back(scalerrecordraw); };
  uint32_t getRunNUmber() { return mRunNumber; };
  int printRates();
  int printIntegrals();
  int printInputRateAndIntegral(int inp);
  int printClassBRateAndIntegralII(int icls);
  int printClassBRateAndIntegral(int iclsinscalers);
  //
  int addOrbitOffset(uint32_t offset);
  //
  // static constexpr uint32_t NCOUNTERS = 1052;
  // v1
  // static constexpr uint32_t NCOUNTERS = 1070;
  // v2 - orbitid added at the end
  static constexpr uint32_t NCOUNTERSv2 = 1071;
  static constexpr uint32_t NCOUNTERS = 1085;
  static std::vector<std::string> scalerNames;

  void printLMBRateVsT() const; // prints LMB interaction rate vs time for debugging

  // returns the pair of global (levelled) interaction rate, as well as interpolated
  // rate in Hz at a certain orbit number within the run
  std::pair<double, double> getRate(uint32_t orbit, int classindex, int type) const;

  /// same with absolute  timestamp (not orbit) as argument
  std::pair<double, double> getRateGivenT(double timestamp, int classindex, int type) const;

  /// retrieves time boundaries of this scaler object from O2 scalers
  std::pair<unsigned long, unsigned long> getTimeLimit() const
  {
    return std::make_pair((unsigned long)mScalerRecordO2[0].epochTime * 1000, (unsigned long)mScalerRecordO2[mScalerRecordO2.size() - 1].epochTime * 1000);
  }
  /// retrieves time boundaries of this scaler object from Raw: should be same as from O2 and can be used without convertRawToO2 call
  std::pair<unsigned long, unsigned long> getTimeLimitFromRaw() const
  {
    return std::make_pair((unsigned long)mScalerRecordRaw[0].epochTime * 1000, (unsigned long)mScalerRecordRaw[mScalerRecordRaw.size() - 1].epochTime * 1000);
  }
  /// retrieves orbit boundaries of this scaler object from O2
  std::pair<unsigned long, unsigned long> getOrbitLimit() const
  {
    return std::make_pair((unsigned long)mScalerRecordO2[0].intRecord.orbit, (unsigned long)mScalerRecordO2[mScalerRecordO2.size() - 1].intRecord.orbit);
  }
  /// retrieves orbit boundaries of this scaler object from Raw: should be same as from O2 and can be used without convertRawToO2 call
  std::pair<unsigned long, unsigned long> getOrbitLimitFromRaw() const
  {
    return std::make_pair((unsigned long)mScalerRecordRaw[0].intRecord.orbit, (unsigned long)mScalerRecordRaw[mScalerRecordRaw.size() - 1].intRecord.orbit);
  }

 private:
  // map from class index to overflow
  // overflow counts how many time class scalerers overflowed
  typedef std::map<uint32_t, std::array<uint32_t, 6>> overflows_t;
  int mVersion = 0;
  uint32_t mRunNumber = 0;
  // using class mask for all class index related stuff
  std::bitset<CTP_NCLASSES> mClassMask;
  o2::detectors::DetID::mask_t mDetectorMask;
  std::vector<CTPScalerRecordRaw> mScalerRecordRaw;
  std::vector<CTPScalerRecordO2> mScalerRecordO2;
  int processScalerLine(const std::string& line, int& level, uint32_t& nclasses);
  int copyRawToO2ScalerRecord(const CTPScalerRecordRaw& rawrec, CTPScalerRecordO2& o2rec, overflows_t& classesoverflows, std::array<uint32_t, 48>& overflows);
  int updateOverflows(const CTPScalerRecordRaw& rec0, const CTPScalerRecordRaw& rec1, overflows_t& classesoverflows) const;
  int updateOverflows(const CTPScalerRaw& scal0, const CTPScalerRaw& scal1, std::array<uint32_t, 6>& overflow) const;
  int updateOverflowsInps(const CTPScalerRecordRaw& rec0, const CTPScalerRecordRaw& rec1, std::array<uint32_t, 48>& overflow) const;
  ClassDefNV(CTPRunScalers, 2);
};
} // namespace ctp
} // namespace o2
#endif //_CTP_SCALERS_H
