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
#include "CommonUtils/StringUtils.h"
#include "FairLogger.h"

using namespace o2::ctp;

void CTPScalerRaw::printStream(std::ostream& stream) const
{
  stream << "Class:" << classIndex << " RAW LMB:" << lmBefore << " LMA:" << lmAfter;
  stream << " LOB:" << l0Before << " L0A:" << l0After;
  stream << " L1B:" << l1Before << " L1A:" << l1After << std::endl;
}
//
void CTPScalerO2::createCTPScalerO2FromRaw(CTPScalerRaw& raw, std::vector<uint32_t>& overflow)
{
  classIndex = raw.classIndex;
  lmBefore = raw.lmBefore + 0xffffffff*overflow[0];
  lmAfter = raw.lmAfter + 0xffffffff*overflow[1];
  l0Before = raw.l0Before + 0xffffffff*overflow[2];
  l0After = raw.l0After + 0xffffffff*overflow[3];
  l1Before = raw.l1Before + 0xffffffff*overflow[4];
  l1After = raw.l1After + 0xffffffff*overflow[5];
}
void CTPScalerO2::printStream(std::ostream& stream) const
{
  stream << "Class:" << classIndex << "O2 LMB:" << lmBefore << " LMA:" << lmAfter;
  stream << " LOB:" << l0Before << " L0A:" << l0After;
  stream << " L1B:" << l1Before << " L1A:" << l1After << std::endl;
}
void CTPScalerRecordRaw::printStream(std::ostream& stream) const
{
  stream << "Orbit:" << intRecord.orbit << " BC:" << intRecord.bc;
  stream << " Seconds:" << seconds << " Microseconds:" << microSeconds << std::endl;
  for(auto const& cnts: scalers) {
      cnts.printStream(stream);
  }
}
void CTPScalerRecordO2::printStream(std::ostream& stream) const
{
  stream << "Orbit:" << intRecord.orbit << " BC:" << intRecord.bc;
  stream << " Seconds:" << seconds << " Microseconds:" << microSeconds << std::endl;
  for(auto const& cnts: scalers) {
      cnts.printStream(stream);
  }
}
void CTPRunScalers::printStream(std::ostream& stream) const
{
  stream << "CTP Scalers (version:" << mVersion << ") Run:" << mRunNumber << std::endl;
  printClasses(stream);
  for(auto const& rec: mScalerRecordRaw) {
      rec.printStream(stream);
  }
}
void CTPRunScalers::printClasses(std::ostream& stream) const
{
  stream << "CTP classes:";
  for(int i=0;i<mClassMask.size(); i++) {
      if(mClassMask[i]) {
          stream << " " << i;
      }
  }
  stream << std::endl;
}
std::vector<uint32_t> CTPRunScalers::getClassIndexes()
{
  std::vector<uint32_t> indexes;
  for(uint32_t i = 0; i < CTP_NCLASSES; i++) {
      if(mClassMask[i]) {
          indexes.push_back(i);
      }
  }
  return indexes;
}
int CTPRunScalers::readScalers(const std::string& rawscalers)
{
  LOG(INFO) << "Loading CTP scalers.";
  std::istringstream iss(rawscalers);
  int ret = 0;
  int level = 0;
  std::string line;
  int nclasses=0;
  while (std::getline(iss, line)) {
    o2::utils::Str::trim(line);
    if ((ret = processScalerLine(line, level, nclasses)) != 0) {
      return ret;
    }
  }
  if(nclasses != 0) {
      LOG(ERROR) << "Wrong number of classes in final record";
      return 6;
  }
  return 0;
}
int CTPRunScalers::processScalerLine(std::string& line, int& level, int& nclasses)
{
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
  std::cout << line << " level in::" << level << std::endl;
  // Version
  if(level == 0) {
      if(ntokens != 1) {
          LOG(ERROR) << "Expected version in the first line";
          return 1;
      } else {
          mVersion = std::stoi(tokens[0]);
          level = 1;
          return 0;
      }
  }
  // Run Number N Classes [class indexes]
  if(level == 1) {
      if(ntokens < 3) {
          LOG(ERROR) << "Wrong syntax of second line in CTP scalers";
          return 2;
      } else {
          mRunNumber = std::stol(tokens[0]);
          int numofclasses = std::stoi(tokens[1]);
          if((numofclasses+2) != ntokens) {
            LOG(ERROR) << "Wrong syntax of second line in CTP scalers";
            return 3;
          }
          mClassMask.reset();
          for(int i=0; i < numofclasses; i++) {
              int index = std::stoi(tokens[i+2]);
              mClassMask[index] = 1;
          }
          level = 2;
          nclasses = 0;
          return 0;
      }
  }
  // Time stamp: orbit bcid linuxtime
  if(level == 2) {
      if(ntokens != 4) {
        LOG(ERROR) << "Wrong syntax of time stamp line in CTP scalers";
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
  if(level == 3) {
    if(ntokens != 6) {
        LOG(ERROR) << "Wrong syntax of counters line in CTP scalers";
        return 5;
      } else {
        std::cout << "nclasses:" << nclasses << std::endl;
        CTPScalerRaw scaler;
        scaler.classIndex = getClassIndexes()[level - 3];
        scaler.lmBefore = std::stol(tokens[0]);
        scaler.lmAfter = std::stol(tokens[1]);
        scaler.l0Before = std::stol(tokens[2]);
        scaler.l0After = std::stol(tokens[3]);
        scaler.l1Before = std::stol(tokens[4]);
        scaler.l1After = std::stol(tokens[5]);
        (mScalerRecordRaw.back()).scalers.push_back(scaler);
        nclasses++;
        if(nclasses >= mClassMask.count()) {
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
  for(uint32_t i = 0; i < mClassMask.size(); i++) {
      if(mClassMask[i]) {
          overflows[i] = {0,0,0,0,0,0};
      }
  }
  CTPScalerRecordO2 o2rec;
  o2rec.intRecord = mScalerRecordRaw[0].intRecord;
  o2rec.seconds = mScalerRecordRaw[0].seconds;
  o2rec.microSeconds = mScalerRecordRaw[0].microSeconds;
  for(int i = 1; i < mScalerRecordRaw.size(); i++) {
    //CTPScalerRecordRaw& prev = mScalerRecordRaw[i-1];
    //CTPScalerRecordRaw& curr = mScalerRecordRaw[i];
  }
  return 0;
}
