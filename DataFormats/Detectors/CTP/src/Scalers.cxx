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
  stream << " miliSeconds:" << epochTime << std::endl;
  for (auto const& cnts : scalers) {
    cnts.printStream(stream);
  }
  for (auto const& dets : scalersDets) {
    stream << dets << " ";
  }
  stream << std::endl;
}
void CTPScalerRecordO2::printStream(std::ostream& stream) const
{
  stream << "Orbit:" << intRecord.orbit << " BC:" << intRecord.bc;
  stream << " miliSeconds:" << epochTime << std::endl;
  for (auto const& cnts : scalers) {
    cnts.printStream(stream);
  }
  for (auto const& dets : scalersDets) {
    stream << dets << " ";
  }
  stream << std::endl;
}
//
// CTPRunScalers
//
void CTPRunScalers::printStream(std::ostream& stream) const
{
  stream << "CTP Scalers (version:" << mVersion << ") Run:" << mRunNumber << std::endl;
  printClasses(stream);
  stream << "Detector mask:" << mDetectorMask << std::endl;
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
      // rec.seconds = std::stol(tokens[2]);
      rec.epochTime = std::stoll(tokens[2]);
      // rec.microSeconds = std::stol(tokens[3]);
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
  // o2rec.seconds = rawrec.seconds;
  // o2rec.microSeconds = rawrec.microSeconds;
  o2rec.epochTime = rawrec.epochTime;
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
  // broken for classes started at L0
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
    LOG(warning) << "L0B > LMA ok if L0 class.";
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
int CTPRunScalers::printRates()
{
  if (mScalerRecordO2.size() == 0) {
    LOG(info) << "ScalerRecord is empty, doing nothing";
    return 0;
  }
  LOG(info) << "Scaler rates for run:" << mRunNumber;
  CTPScalerRecordO2* scalrec0 = &mScalerRecordO2[0];
  for (int i = 1; i < mScalerRecordO2.size(); i++) {
    CTPScalerRecordO2* scalrec1 = &mScalerRecordO2[i];
    double_t tt = (double_t)(scalrec1->intRecord.orbit - scalrec0->intRecord.orbit);
    tt = tt * 88e-6;
    for (int j = 0; j < scalrec1->scalers.size(); j++) {
      CTPScalerO2* s0 = &(scalrec0->scalers[j]);
      CTPScalerO2* s1 = &(scalrec1->scalers[j]);
      double_t rMB = (s1->lmBefore - s0->lmBefore) / tt;
      double_t rMA = (s1->lmAfter - s0->lmAfter) / tt;
      double_t r0B = (s1->l0Before - s0->l0Before) / tt;
      double_t r0A = (s1->l0After - s0->l0After) / tt;
      double_t r1B = (s1->l1Before - s0->l1Before) / tt;
      double_t r1A = (s1->l1After - s0->l1After) / tt;
      std::cout << "Class:" << s0->classIndex << ": ";
      std::cout << rMB << "  " << rMA << "  ";
      std::cout << r0B << "  " << r0A << "  ";
      std::cout << r1B << "   " << r1A;
      std::cout << std::endl;
    }
    scalrec0 = scalrec1;
  }
  return 0;
}
int CTPRunScalers::printIntegrals()
{
  if (mScalerRecordO2.size() == 0) {
    LOG(info) << "ScalerRecord is empty, doing nothing";
    return 0;
  }
  LOG(info) << "Scaler Integrals for run:" << mRunNumber;

  for (int i = 0; i < mScalerRecordO2[0].scalers.size(); i++) {
    std::cout << i << " LMB " << mScalerRecordO2[mScalerRecordO2.size() - 1].scalers[i].lmBefore - mScalerRecordO2[0].scalers[i].lmBefore << std::endl;
    std::cout << i << " LMA " << mScalerRecordO2[mScalerRecordO2.size() - 1].scalers[i].lmAfter - mScalerRecordO2[0].scalers[i].lmAfter << std::endl;
    std::cout << i << " L0B " << mScalerRecordO2[mScalerRecordO2.size() - 1].scalers[i].l0Before - mScalerRecordO2[0].scalers[i].l0Before << std::endl;
    std::cout << i << " L0A " << mScalerRecordO2[mScalerRecordO2.size() - 1].scalers[i].l0After - mScalerRecordO2[0].scalers[i].l0After << std::endl;
    std::cout << i << " L1B " << mScalerRecordO2[mScalerRecordO2.size() - 1].scalers[i].l1Before - mScalerRecordO2[0].scalers[i].l1Before << std::endl;
    std::cout << i << " L1A " << mScalerRecordO2[mScalerRecordO2.size() - 1].scalers[i].l1After - mScalerRecordO2[0].scalers[i].l1After << std::endl;
  }
  return 0;
}
std::vector<std::string> CTPRunScalers::scalerNames =
  {
    "runn0", "runn1", "runn2", "runn3", "runn4", "runn5", "runn6", "runn7", "runn8", "runn9", "runn10", "runn11", "runn12", "runn13", "runn14", "runn15", "ltg1_ORB", "ltg1_HB", "ltg1_HBr", "ltg1_HC", "ltg1_PH", "ltg1_PP", "ltg1_CAL", "ltg1_SOT", "ltg1_EOT", "ltg1_SOC", "ltg1_EOC", "ltg1_TF", "ltg1_FERST", "ltg1_RT", "ltg1_RS", "", "", "", "", "", "", "", "", "", "", "", "", "ltg1_GAP1", "ltg1_GAP2", "ltg1_TPC_sync", "ltg1_TPC_rst", "ltg1_TOF", "ltg2_ORB", "ltg2_HB", "ltg2_HBr", "ltg2_HC", "ltg2_PH", "ltg2_PP", "ltg2_CAL", "ltg2_SOT", "ltg2_EOT", "ltg2_SOC", "ltg2_EOC", "ltg2_TF", "ltg2_FERST", "ltg2_RT", "ltg2_RS", "", "", "", "", "", "", "", "", "", "", "", "", "ltg2_GAP1", "ltg2_GAP2", "ltg2_TPC_sync", "ltg2_TPC_rst", "ltg2_TOF", "ltg3_ORB", "ltg3_HB", "ltg3_HBr", "ltg3_HC", "ltg3_PH", "ltg3_PP", "ltg3_CAL", "ltg3_SOT", "ltg3_EOT", "ltg3_SOC", "ltg3_EOC", "ltg3_TF", "ltg3_FERST", "ltg3_RT", "ltg3_RS", "", "", "", "", "", "", "", "", "", "", "", "", "ltg3_GAP1", "ltg3_GAP2", "ltg3_TPC_sync", "ltg3_TPC_rst", "ltg3_TOF", "ltg4_ORB", "ltg4_HB", "ltg4_HBr", "ltg4_HC", "ltg4_PH", "ltg4_PP", "ltg4_CAL", "ltg4_SOT", "ltg4_EOT", "ltg4_SOC", "ltg4_EOC", "ltg4_TF", "ltg4_FERST", "ltg4_RT", "ltg4_RS", "", "", "", "", "", "", "", "", "", "", "", "", "ltg4_GAP1", "ltg4_GAP2", "ltg4_TPC_sync", "ltg4_TPC_rst", "ltg4_TOF", "ltg5_ORB", "ltg5_HB", "ltg5_HBr", "ltg5_HC", "ltg5_PH", "ltg5_PP", "ltg5_CAL", "ltg5_SOT", "ltg5_EOT", "ltg5_SOC", "ltg5_EOC", "ltg5_TF", "ltg5_FERST", "ltg5_RT", "ltg5_RS", "", "", "", "", "", "", "", "", "", "", "", "", "ltg5_GAP1", "ltg5_GAP2", "ltg5_TPC_sync", "ltg5_TPC_rst", "ltg5_TOF", "ltg6_ORB", "ltg6_HB", "ltg6_HBr", "ltg6_HC", "ltg6_PH", "ltg6_PP", "ltg6_CAL", "ltg6_SOT", "ltg6_EOT", "ltg6_SOC", "ltg6_EOC", "ltg6_TF", "ltg6_FERST", "ltg6_RT", "ltg6_RS", "", "", "", "", "", "", "", "", "", "", "", "", "ltg6_GAP1", "ltg6_GAP2", "ltg6_TPC_sync", "ltg6_TPC_rst", "ltg6_TOF", "ltg7_ORB", "ltg7_HB", "ltg7_HBr", "ltg7_HC", "ltg7_PH", "ltg7_PP", "ltg7_CAL", "ltg7_SOT", "ltg7_EOT", "ltg7_SOC", "ltg7_EOC", "ltg7_TF", "ltg7_FERST", "ltg7_RT", "ltg7_RS", "", "", "", "", "", "", "", "", "", "", "", "", "ltg7_GAP1", "ltg7_GAP2", "ltg7_TPC_sync", "ltg7_TPC_rst", "ltg7_TOF", "ltg8_ORB", "ltg8_HB", "ltg8_HBr", "ltg8_HC", "ltg8_PH", "ltg8_PP", "ltg8_CAL", "ltg8_SOT", "ltg8_EOT", "ltg8_SOC", "ltg8_EOC", "ltg8_TF", "ltg8_FERST", "ltg8_RT", "ltg8_RS", "", "", "", "", "", "", "", "", "", "", "", "", "ltg8_GAP1", "ltg8_GAP2", "ltg8_TPC_sync", "ltg8_TPC_rst", "ltg8_TOF", "ltg9_ORB", "ltg9_HB", "ltg9_HBr", "ltg9_HC", "ltg9_PH", "ltg9_PP", "ltg9_CAL", "ltg9_SOT", "ltg9_EOT", "ltg9_SOC", "ltg9_EOC", "ltg9_TF", "ltg9_FERST", "ltg9_RT", "ltg9_RS", "", "", "", "", "", "", "", "", "", "", "", "", "ltg9_GAP1", "ltg9_GAP2", "ltg9_TPC_sync", "ltg9_TPC_rst", "ltg9_TOF", "ltg10_ORB", "ltg10_HB", "ltg10_HBr", "ltg10_HC", "ltg10_PH", "ltg10_PP", "ltg10_CAL", "ltg10_SOT", "ltg10_EOT", "ltg10_SOC", "ltg10_EOC", "ltg10_TF", "ltg10_FERST", "ltg10_RT", "ltg10_RS", "", "", "", "", "", "", "", "", "", "", "", "", "ltg10_GAP1", "ltg10_GAP2", "ltg10_TPC_sync", "ltg10_TPC_rst", "ltg10_TOF", "ltg11_ORB", "ltg11_HB", "ltg11_HBr", "ltg11_HC", "ltg11_PH", "ltg11_PP", "ltg11_CAL", "ltg11_SOT", "ltg11_EOT", "ltg11_SOC", "ltg11_EOC", "ltg11_TF", "ltg11_FERST", "ltg11_RT", "ltg11_RS", "", "", "", "", "", "", "", "", "", "", "", "", "ltg11_GAP1", "ltg11_GAP2", "ltg11_TPC_sync", "ltg11_TPC_rst", "ltg11_TOF", "ltg12_ORB", "ltg12_HB", "ltg12_HBr", "ltg12_HC", "ltg12_PH", "ltg12_PP", "ltg12_CAL", "ltg12_SOT", "ltg12_EOT", "ltg12_SOC", "ltg12_EOC", "ltg12_TF", "ltg12_FERST", "ltg12_RT", "ltg12_RS", "", "", "", "", "", "", "", "", "", "", "", "", "ltg12_GAP1", "ltg12_GAP2", "ltg12_TPC_sync", "ltg12_TPC_rst", "ltg12_TOF", "ltg13_ORB", "ltg13_HB", "ltg13_HBr", "ltg13_HC", "ltg13_PH", "ltg13_PP", "ltg13_CAL", "ltg13_SOT", "ltg13_EOT", "ltg13_SOC", "ltg13_EOC", "ltg13_TF", "ltg13_FERST", "ltg13_RT", "ltg13_RS", "", "", "", "", "", "", "", "", "", "", "", "", "ltg13_GAP1", "ltg13_GAP2", "ltg13_TPC_sync", "ltg13_TPC_rst", "ltg13_TOF", "ltg14_ORB", "ltg14_HB", "ltg14_HBr", "ltg14_HC", "ltg14_PH", "ltg14_PP", "ltg14_CAL", "ltg14_SOT", "ltg14_EOT", "ltg14_SOC", "ltg14_EOC", "ltg14_TF", "ltg14_FERST", "ltg14_RT", "ltg14_RS", "", "", "", "", "", "", "", "", "", "", "", "", "ltg14_GAP1", "ltg14_GAP2", "ltg14_TPC_sync", "ltg14_TPC_rst", "ltg14_TOF", "ltg15_ORB", "ltg15_HB", "ltg15_HBr", "ltg15_HC", "ltg15_PH", "ltg15_PP", "ltg15_CAL", "ltg15_SOT", "ltg15_EOT", "ltg15_SOC", "ltg15_EOC", "ltg15_TF", "ltg15_FERST", "ltg15_RT", "ltg15_RS", "", "", "", "", "", "", "", "", "", "", "", "", "ltg15_GAP1", "ltg15_GAP2", "ltg15_TPC_sync", "ltg15_TPC_rst", "ltg15_TOF", "ltg16_ORB", "ltg16_HB", "ltg16_HBr", "ltg16_HC", "ltg16_PH", "ltg16_PP", "ltg16_CAL", "ltg16_SOT", "ltg16_EOT", "ltg16_SOC", "ltg16_EOC", "ltg16_TF", "ltg16_FERST", "ltg16_RT", "ltg16_RS", "", "", "", "", "", "", "", "", "", "", "", "", "ltg16_GAP1", "ltg16_GAP2", "ltg16_TPC_sync", "ltg16_TPC_rst", "ltg16_TOF", "ltg17_ORB", "ltg17_HB", "ltg17_HBr", "ltg17_HC", "ltg17_PH", "ltg17_PP", "ltg17_CAL", "ltg17_SOT", "ltg17_EOT", "ltg17_SOC", "ltg17_EOC", "ltg17_TF", "ltg17_FERST", "ltg17_RT", "ltg17_RS", "", "", "", "", "", "", "", "", "", "", "", "", "ltg17_GAP1", "ltg17_GAP2", "ltg17_TPC_sync", "ltg17_TPC_rst", "ltg17_TOF", "ltg18_ORB", "ltg18_HB", "ltg18_HBr", "ltg18_HC", "ltg18_PH", "ltg18_PP", "ltg18_CAL", "ltg18_SOT", "ltg18_EOT", "ltg18_SOC", "ltg18_EOC", "ltg18_TF", "ltg18_FERST", "ltg18_RT", "ltg18_RS", "", "", "", "", "", "", "", "", "", "", "", "", "ltg18_GAP1", "ltg18_GAP2", "ltg18_TPC_sync", "ltg18_TPC_rst", "ltg18_TOF", "bc40", "clk240", "extorb", "PLSRin", "FastLMin", "BUSYin", "SPAREin", "inp1", "inp2", "inp3", "inp4", "inp5", "inp6", "inp7", "inp8", "inp9", "inp10", "inp11", "inp12", "inp13", "inp14", "inp15", "inp16", "inp17", "inp18", "inp19", "inp20", "inp21", "inp22", "inp23", "inp24", "inp25", "inp26", "inp27", "inp28", "inp29", "inp30", "inp31", "inp32", "inp33", "inp34", "inp35", "inp36", "inp37", "inp38", "inp39", "inp40", "inp41", "inp42", "inp43", "inp44", "inp45", "inp46", "inp47", "inp48", "clamb1", "clamb2", "clamb3", "clamb4", "clamb5", "clamb6", "clamb7", "clamb8", "clamb9", "clamb10", "clamb11", "clamb12", "clamb13", "clamb14", "clamb15", "clamb16", "clamb17", "clamb18", "clamb19", "clamb20", "clamb21", "clamb22", "clamb23", "clamb24", "clamb25", "clamb26", "clamb27", "clamb28", "clamb29", "clamb30", "clamb31", "clamb32", "clamb33", "clamb34", "clamb35", "clamb36", "clamb37", "clamb38", "clamb39", "clamb40", "clamb41", "clamb42", "clamb43", "clamb44", "clamb45", "clamb46", "clamb47", "clamb48", "clamb49", "clamb50", "clamb51", "clamb52", "clamb53", "clamb54", "clamb55", "clamb56", "clamb57", "clamb58", "clamb59", "clamb60", "clamb61", "clamb62", "clamb63", "clamb64", "clama1", "clama2", "clama3", "clama4", "clama5", "clama6", "clama7", "clama8", "clama9", "clama10", "clama11", "clama12", "clama13", "clama14", "clama15", "clama16", "clama17", "clama18", "clama19", "clama20", "clama21", "clama22", "clama23", "clama24", "clama25", "clama26", "clama27", "clama28", "clama29", "clama30", "clama31", "clama32", "clama33", "clama34", "clama35", "clama36", "clama37", "clama38", "clama39", "clama40", "clama41", "clama42", "clama43", "clama44", "clama45", "clama46", "clama47", "clama48", "clama49", "clama50", "clama51", "clama52", "clama53", "clama54", "clama55", "clama56", "clama57", "clama58", "clama59", "clama60", "clama61", "clama62", "clama63", "clama64", "cla0b1", "cla0b2", "cla0b3", "cla0b4", "cla0b5", "cla0b6", "cla0b7", "cla0b8", "cla0b9", "cla0b10", "cla0b11", "cla0b12", "cla0b13", "cla0b14", "cla0b15", "cla0b16", "cla0b17", "cla0b18", "cla0b19", "cla0b20", "cla0b21", "cla0b22", "cla0b23", "cla0b24", "cla0b25", "cla0b26", "cla0b27", "cla0b28", "cla0b29", "cla0b30", "cla0b31", "cla0b32", "cla0b33", "cla0b34", "cla0b35", "cla0b36", "cla0b37", "cla0b38", "cla0b39", "cla0b40", "cla0b41", "cla0b42", "cla0b43", "cla0b44", "cla0b45", "cla0b46", "cla0b47", "cla0b48", "cla0b49", "cla0b50", "cla0b51", "cla0b52", "cla0b53", "cla0b54", "cla0b55", "cla0b56", "cla0b57", "cla0b58", "cla0b59", "cla0b60", "cla0b61", "cla0b62", "cla0b63", "cla0b64", "cla0a1", "cla0a2", "cla0a3", "cla0a4", "cla0a5", "cla0a6", "cla0a7", "cla0a8", "cla0a9", "cla0a10", "cla0a11", "cla0a12", "cla0a13", "cla0a14", "cla0a15", "cla0a16", "cla0a17", "cla0a18", "cla0a19", "cla0a20", "cla0a21", "cla0a22", "cla0a23", "cla0a24", "cla0a25", "cla0a26", "cla0a27", "cla0a28", "cla0a29", "cla0a30", "cla0a31", "cla0a32", "cla0a33", "cla0a34", "cla0a35", "cla0a36", "cla0a37", "cla0a38", "cla0a39", "cla0a40", "cla0a41", "cla0a42", "cla0a43", "cla0a44", "cla0a45", "cla0a46", "cla0a47", "cla0a48", "cla0a49", "cla0a50", "cla0a51", "cla0a52", "cla0a53", "cla0a54", "cla0a55", "cla0a56", "cla0a57", "cla0a58", "cla0a59", "cla0a60", "cla0a61", "cla0a62", "cla0a63", "cla0a64", "cla1b1", "cla1b2", "cla1b3", "cla1b4", "cla1b5", "cla1b6", "cla1b7", "cla1b8", "cla1b9", "cla1b10", "cla1b11", "cla1b12", "cla1b13", "cla1b14", "cla1b15", "cla1b16", "cla1b17", "cla1b18", "cla1b19", "cla1b20", "cla1b21", "cla1b22", "cla1b23", "cla1b24", "cla1b25", "cla1b26", "cla1b27", "cla1b28", "cla1b29", "cla1b30", "cla1b31", "cla1b32", "cla1b33", "cla1b34", "cla1b35", "cla1b36", "cla1b37", "cla1b38", "cla1b39", "cla1b40", "cla1b41", "cla1b42", "cla1b43", "cla1b44", "cla1b45", "cla1b46", "cla1b47", "cla1b48", "cla1b49", "cla1b50", "cla1b51", "cla1b52", "cla1b53", "cla1b54", "cla1b55", "cla1b56", "cla1b57", "cla1b58", "cla1b59", "cla1b60", "cla1b61", "cla1b62", "cla1b63", "cla1b64", "cla1a1", "cla1a2", "cla1a3", "cla1a4", "cla1a5", "cla1a6", "cla1a7", "cla1a8", "cla1a9", "cla1a10", "cla1a11", "cla1a12", "cla1a13", "cla1a14", "cla1a15", "cla1a16", "cla1a17", "cla1a18", "cla1a19", "cla1a20", "cla1a21", "cla1a22", "cla1a23", "cla1a24", "cla1a25", "cla1a26", "cla1a27", "cla1a28", "cla1a29", "cla1a30", "cla1a31", "cla1a32", "cla1a33", "cla1a34", "cla1a35", "cla1a36", "cla1a37", "cla1a38", "cla1a39", "cla1a40", "cla1a41", "cla1a42", "cla1a43", "cla1a44", "cla1a45", "cla1a46", "cla1a47", "cla1a48", "cla1a49", "cla1a50", "cla1a51", "cla1a52", "cla1a53", "cla1a54", "cla1a55", "cla1a56", "cla1a57", "cla1a58", "cla1a59", "cla1a60", "cla1a61", "cla1a62", "cla1a63", "cla1a64", "l0_trigger", "l1_trigger", "l2_trigger", "clum1", "clum2", "clum3", "clum4", "clum5", "clum6", "clu01", "clu02", "clu03", "clu04", "clu05", "clu06", "clu11", "clu12", "clu13", "clu14", "clu15", "clu16",
    "ltg1_busy", "ltg2_busy", "ltg3_busy", "ltg4_busy", "ltg5_busy", "ltg6_busy", "ltg7_busy", "ltg8_busy", "ltg9_busy",
    "ltg10_busy", "ltg11_busy", "ltg12_busy", "ltg13_busy", "ltg14_busy", "ltg15_busy", "ltg16_busy", "ltg17_busy", "ltg18_busy"};
