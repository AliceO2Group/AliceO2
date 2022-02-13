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

/// \file Scalers.cxx
/// \author Roman Lietava

#include "DataFormatsCTP/Scalers.h"
#include <iostream>
#include <iomanip>
#include "CommonUtils/StringUtils.h"
#include "FairLogger.h"

using namespace o2::ctp;

void CTPScalerRaw::printStream(std::ostream& stream) const
{
  stream << "Class:" << std::setw(2) << classIndex << " RAW";
  stream << " LMB:" << std::setw(10) << lmBefore << " LMA:" << std::setw(10) << lmAfter;
  stream << " LOB:" << std::setw(10) << l0Before << " L0A:" << std::setw(10) << l0After;
  stream << " L1B:" << std::setw(10) << l1Before << " L1A:" << std::setw(10) << l1After << std::endl;
}
//
void CTPScalerO2::createCTPScalerO2FromRaw(const CTPScalerRaw& raw, const std::array<uint32_t, 6>& overflow)
{
  classIndex = raw.classIndex;
  lmBefore = (uint64_t)(raw.lmBefore) + 0xffffffffull * (uint64_t)(overflow[0]);
  lmAfter = (uint64_t)(raw.lmAfter) + 0xffffffffull * (uint64_t)(overflow[1]);
  l0Before = (uint64_t)(raw.l0Before) + 0xffffffffull * (uint64_t)(overflow[2]);
  l0After = (uint64_t)(raw.l0After) + 0xffffffffull * (uint64_t)(overflow[3]);
  l1Before = (uint64_t)(raw.l1Before) + 0xffffffffull * (uint64_t)(overflow[4]);
  l1After = (uint64_t)(raw.l1After) + 0xffffffffull * (uint64_t)(overflow[5]);
  //std::cout << "lmb overflow:" << overflow[0] << " lmb:" << lmBefore << " raw:" << raw.lmBefore << std::endl;
}
void CTPScalerO2::printStream(std::ostream& stream) const
{
  stream << "Class:" << std::setw(2) << classIndex << " O2";
  stream << " LMB:" << std::setw(10) << lmBefore << " LMA:" << std::setw(10) << lmAfter;
  stream << " LOB:" << std::setw(10) << l0Before << " L0A:" << std::setw(10) << l0After;
  stream << " L1B:" << std::setw(10) << l1Before << " L1A:" << std::setw(10) << l1After << std::endl;
}
void CTPScalerRecordRaw::printStream(std::ostream& stream) const
{
  stream << "Orbit:" << intRecord.orbit << " BC:" << intRecord.bc;
  stream << " Seconds:" << seconds << " Microseconds:" << microSeconds << std::endl;
  for (auto const& cnts : scalers) {
    cnts.printStream(stream);
  }
}
void CTPScalerRecordO2::printStream(std::ostream& stream) const
{
  stream << "Orbit:" << intRecord.orbit << " BC:" << intRecord.bc;
  stream << " Seconds:" << seconds << " Microseconds:" << microSeconds << std::endl;
  for (auto const& cnts : scalers) {
    cnts.printStream(stream);
  }
}
//
// CTPRunScalers
//
void CTPRunScalers::printStream(std::ostream& stream) const
{
  stream << "CTP Scalers (version:" << mVersion << ") Run:" << mRunNumber << std::endl;
  printClasses(stream);
  for (auto const& rec : mScalerRecordRaw) {
    rec.printStream(stream);
  }
  stream << "O2 Counters:" << std::endl;
  for (auto const& rec : mScalerRecordO2) {
    rec.printStream(stream);
  }
}
void CTPRunScalers::printClasses(std::ostream& stream) const
{
  stream << "CTP classes:";
  for (int i = 0; i < mClassMask.size(); i++) {
    if (mClassMask[i]) {
      stream << " " << i;
    }
  }
  stream << std::endl;
}
std::vector<uint32_t> CTPRunScalers::getClassIndexes() const
{
  std::vector<uint32_t> indexes;
  for (uint32_t i = 0; i < CTP_NCLASSES; i++) {
    if (mClassMask[i]) {
      indexes.push_back(i);
    }
  }
  return indexes;
}
int CTPRunScalers::readScalers(const std::string& rawscalers)
{
  LOG(info) << "Loading CTP scalers.";
  std::istringstream iss(rawscalers);
  int ret = 0;
  int level = 0;
  std::string line;
  int nclasses = 0;
  int nlines = 0;
  while (std::getline(iss, line)) {
    o2::utils::Str::trim(line);
    if ((ret = processScalerLine(line, level, nclasses)) != 0) {
      return ret;
    }
    nlines++;
  }
  if (nlines < 4) {
    LOG(error) << "Input string seems too small:\n"
               << rawscalers;
    return 6;
  }
  if (nclasses != 0) {
    LOG(error) << "Wrong number of classes in final record";
    return 6;
  }
  return 0;
}
int CTPRunScalers::processScalerLine(const std::string& line, int& level, int& nclasses)
{
  //std::cout << "Processing line" << std::endl;
  if (line.size() == 0) {
    return 0;
  }
  if (line.at(0) == '#') {
    return 0;
  }
  std::vector<std::string> tokens = o2::utils::Str::tokenize(line, ' ');
  size_t ntokens = tokens.size();
  if (ntokens == 0) {
    return 0;
  }
  //std::cout << line << " level in::" << level << std::endl;
  // Version
  if (level == 0) {
    if (ntokens != 1) {
      LOG(error) << "Expected version in the first line";
      return 1;
    } else {
      mVersion = std::stoi(tokens[0]);
      level = 1;
      return 0;
    }
  }
  // Run Number N Classes [class indexes]
  if (level == 1) {
    if (ntokens < 3) {
      LOG(error) << "Wrong syntax of second line in CTP scalers";
      return 2;
    } else {
      mRunNumber = std::stol(tokens[0]);
      int numofclasses = std::stoi(tokens[1]);
      if ((numofclasses + 2) != ntokens) {
        LOG(error) << "Wrong syntax of second line in CTP scalers";
        return 3;
      }
      mClassMask.reset();
      for (int i = 0; i < numofclasses; i++) {
        int index = std::stoi(tokens[i + 2]);
        mClassMask[index] = 1;
      }
      level = 2;
      nclasses = 0;
      return 0;
    }
  }
  // Time stamp: orbit bcid linuxtime
  if (level == 2) {
    if (ntokens != 4) {
      LOG(error) << "Wrong syntax of time stamp line in CTP scalers";
      return 4;
    } else {
      CTPScalerRecordRaw rec;
      rec.intRecord.orbit = std::stol(tokens[0]);
      rec.intRecord.bc = std::stol(tokens[1]);
      rec.seconds = std::stol(tokens[2]);
      rec.microSeconds = std::stol(tokens[3]);
      mScalerRecordRaw.push_back(rec);
      level = 3;
      return 0;
    }
  }
  if (level == 3) {
    if (ntokens != 6) {
      LOG(error) << "Wrong syntax of counters line in CTP scalers";
      return 5;
    } else {
      //std::cout << "nclasses:" << nclasses << std::endl;
      CTPScalerRaw scaler;
      scaler.classIndex = getClassIndexes()[nclasses];
      scaler.lmBefore = std::stol(tokens[0]);
      scaler.lmAfter = std::stol(tokens[1]);
      scaler.l0Before = std::stol(tokens[2]);
      scaler.l0After = std::stol(tokens[3]);
      scaler.l1Before = std::stol(tokens[4]);
      scaler.l1After = std::stol(tokens[5]);
      (mScalerRecordRaw.back()).scalers.push_back(scaler);
      nclasses++;
      if (nclasses >= mClassMask.count()) {
        level = 2;
        nclasses = 0;
      }
      return 0;
    }
  }
  return 0;
}
/// Converts raw 32 bit scalers to O2 64 bit scalers correcting for overflow.
/// Consistency checks done.
int CTPRunScalers::convertRawToO2()
{
  //struct classScalersOverflows {uint32_t overflows[6];};
  overflows_t overflows;
  for (uint32_t i = 0; i < mClassMask.size(); i++) {
    if (mClassMask[i]) {
      overflows[i] = {0, 0, 0, 0, 0, 0};
    }
  }
  // 1st o2 rec is just copy
  CTPScalerRecordO2 o2rec;
  copyRawToO2ScalerRecord(mScalerRecordRaw[0], o2rec, overflows);
  mScalerRecordO2.push_back(o2rec);
  for (uint32_t i = 1; i < mScalerRecordRaw.size(); i++) {
    //update overflows
    updateOverflows(mScalerRecordRaw[i - 1], mScalerRecordRaw[i], overflows);
    //
    CTPScalerRecordO2 o2rec;
    copyRawToO2ScalerRecord(mScalerRecordRaw[i], o2rec, overflows);
    mScalerRecordO2.push_back(o2rec);
    // Check consistency
    checkConsistency(mScalerRecordO2[i - 1], mScalerRecordO2[i]);
  }
  return 0;
}
int CTPRunScalers::copyRawToO2ScalerRecord(const CTPScalerRecordRaw& rawrec, CTPScalerRecordO2& o2rec, overflows_t& classesoverflows)
{
  if (rawrec.scalers.size() != (mClassMask.count())) {
    LOG(error) << "Inconsistent scaler record size:" << rawrec.scalers.size() << " Expected:" << mClassMask.count();
    return 1;
  }
  o2rec.scalers.clear();
  o2rec.intRecord = rawrec.intRecord;
  o2rec.seconds = rawrec.seconds;
  o2rec.microSeconds = rawrec.microSeconds;
  for (uint32_t i = 0; i < rawrec.scalers.size(); i++) {
    CTPScalerRaw rawscal = rawrec.scalers[i];
    CTPScalerO2 o2scal;
    int k = (getClassIndexes())[i];
    o2scal.createCTPScalerO2FromRaw(rawscal, classesoverflows[k]);
    o2rec.scalers.push_back(o2scal);
  }
  return 0;
}
int CTPRunScalers::checkConsistency(const CTPScalerO2& scal0, const CTPScalerO2& scal1) const
{
  int ret = 0;
  // Scaler should never decrease
  if (scal0.lmBefore > scal1.lmBefore) {
    LOG(error) << "Scaler decreasing: Class:" << scal0.classIndex << " lmBefore 0:" << scal0.lmBefore << " lmBefore :" << scal1.lmBefore;
    ret++;
  }
  if (scal0.l0Before > scal1.l0Before) {
    LOG(error) << "Scaler decreasing: Class:" << scal0.classIndex << " lmBefore 0:" << scal0.l0Before << " lmBefore :" << scal1.l0Before;
    ret++;
  }
  if (scal0.l1Before > scal1.l1Before) {
    LOG(error) << "Scaler decreasing: Class:" << scal0.classIndex << " lmBefore 0:" << scal0.l1Before << " lmBefore :" << scal1.l1Before;
    ret++;
  }
  if (scal0.lmAfter > scal1.lmAfter) {
    LOG(error) << "Scaler decreasing: Class:" << scal0.classIndex << " lmAfter 0:" << scal0.lmAfter << " lmAfter :" << scal1.lmAfter;
    ret++;
  }
  if (scal0.l0After > scal1.l0After) {
    LOG(error) << "Scaler decreasing: Class:" << scal0.classIndex << " lmAfter 0:" << scal0.l0After << " lmAfter :" << scal1.l0After;
    ret++;
  }
  if (scal0.l1After > scal1.l1After) {
    LOG(error) << "Scaler decreasing: Class:" << scal0.classIndex << " lmAfter 0:" << scal0.l1After << " lmAfter :" << scal1.l1After;
    ret++;
  }
  //
  // LMB >= LMA >= L0B >= L0A >= L1B >= L1A: 5 relations
  //
  if ((scal1.lmAfter - scal0.lmAfter) > (scal1.lmBefore - scal0.lmBefore)) {
    LOG(error) << "LMA > LMB eerror";
    ret++;
  }
  if ((scal1.l0After - scal0.l0After) > (scal1.l0Before - scal0.l0Before)) {
    LOG(error) << "L0A > L0B error";
    ret++;
  }
  if ((scal1.l1After - scal0.l1After) > (scal1.l1Before - scal0.l1Before)) {
    LOG(error) << "L1A > L1B error";
    ret++;
  }
  if ((scal1.l0Before - scal0.l0Before) > (scal1.lmAfter - scal0.lmAfter)) {
    LOG(error) << "L0B > LMA error.";
    ret++;
  }
  if ((scal1.l1Before - scal0.l1Before) > (scal1.l0After - scal0.l0After)) {
    LOG(error) << "L1B > L0A Before error.";
    ret++;
  }
  //
  if (ret) {
    scal0.printStream(std::cout);
    scal1.printStream(std::cout);
  }
  return ret;
}
int CTPRunScalers::updateOverflows(const CTPScalerRecordRaw& rec0, const CTPScalerRecordRaw& rec1, overflows_t& classesoverflows) const
{
  if (rec1.scalers.size() != mClassMask.count()) {
    LOG(error) << "Inconsistent scaler record size:" << rec1.scalers.size() << " Expected:" << mClassMask.count();
    return 1;
  }
  for (int i = 0; i < rec0.scalers.size(); i++) {
    int k = (getClassIndexes())[i];
    updateOverflows(rec0.scalers[i], rec1.scalers[i], classesoverflows[k]);
  }
  return 0;
}
int CTPRunScalers::checkConsistency(const CTPScalerRecordO2& rec0, const CTPScalerRecordO2& rec1) const
{
  for (int i = 0; i < rec0.scalers.size(); i++) {
    checkConsistency(rec0.scalers[i], rec1.scalers[i]);
  }
  return 0;
}
int CTPRunScalers::updateOverflows(const CTPScalerRaw& scal0, const CTPScalerRaw& scal1, std::array<uint32_t, 6>& overflow) const
{
  if (scal0.lmBefore > scal1.lmBefore) {
    overflow[0] += 1;
  }
  if (scal0.lmAfter > scal1.lmAfter) {
    overflow[1] += 1;
  }
  if (scal0.l0Before > scal1.l0Before) {
    overflow[2] += 1;
  }
  if (scal0.l0After > scal1.l0After) {
    overflow[3] += 1;
  }
  if (scal0.l1Before > scal1.l1Before) {
    overflow[4] += 1;
  }
  if (scal0.l1After > scal1.l1After) {
    overflow[5] += 1;
  }
  //std::cout << "lmB0:" << scal0.lmBefore << " lmB1:" << scal1.lmBefore << " over:" << overflow[0] << std::endl;
  //for(int i = 0; i < 6; i++)std::cout << overflow[i] << " ";
  //std::cout << std::endl;
  return 0;
}
int CTPRunScalers::parseZMQScalers(std::string zmqscalers)
{
  std::vector<std::string> tokens = o2::utils::Str::tokenize(zmqscalers, ' ');
  return 0;
}
