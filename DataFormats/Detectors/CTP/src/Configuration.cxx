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

/// \file Configuration.cxx
/// \author Roman Lietava

#include "DataFormatsCTP/Configuration.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include <sstream>
#include <regex>
#include "CommonUtils/StringUtils.h"
#include <fairlogger/Logger.h>

using namespace o2::ctp;
//
const std::map<std::string, std::string> CTPInput::run2DetToRun3Det = {{"T", "FT0"}, {"V", "FV0"}, {"U", "FDD"}, {"E", "EMC"}, {"D", "EMC"}, {"H", "TRD"}, {"O", "TOF"}, {"P", "PHS"}, {"Z", "ZDC"}};
const std::map<std::string, std::string> CTPConfiguration::detName2LTG = {{"FV0", "1"}, {"FT0", "2"}, {"FDD", "3"}, {"ITS", "4"}, {"TOF", "5"}, {"MFT", "6"}, {"TPC", "7"}, {"MCH", "8"}, {"MID", "9"}, {"TST", "10"}, {"TRD", "13"}, {"HMP", "14"}, {"ZDC", "15"}, {"PHS", "16"}, {"EMC", "17"}, {"CPV", "18"}};
//
bool CTPConfiguration::isDetector(const o2::detectors::DetID& det)
{
  bool isdet = det.getID() >= det.getNDetectors();
  isdet |= det.getID() < 0;
  if (isdet) {
    LOG(error) << " Detector does not exist: " << det.getID();
    return false;
  }
  return true;
}
void CTPConfiguration::capitaliseString(std::string& str)
{
  for (auto& c : str) {
    c = std::toupper(c);
  }
}
bool CTPConfiguration::isNumber(const std::string& s)
{
  return !s.empty() && std::find_if(s.begin(),
                                    s.end(), [](unsigned char c) { return !std::isdigit(c); }) == s.end();
}
//
int BCMask::setBCmask(std::vector<std::string>& tokens)
{
  BCmask.reset();
  name = tokens[1];
  bool coded = tokens[2].find("L") != std::string::npos;
  coded |= tokens[2].find("H") != std::string::npos;
  std::cout << "coded:" << coded << std::endl;
  if (coded) {
    // jusko notation
    std::string bcmaskstr = tokens[2];
    size_t pos = 0;
    size_t posnext = 0;
    int bccur = 0;
    while (bccur < 3564) {
      // std::cout << "pos:" << pos << std::endl;
      size_t posH = bcmaskstr.find('H', pos);
      size_t posL = bcmaskstr.find('L', pos);
      // std::cout << "H:" << posH << " L:" << posL << std::endl;
      bool b = 1;
      posnext = posH;
      if (posL < posH) {
        posnext = posL;
        b = 0;
      }
      std::string bcsub = bcmaskstr.substr(pos, posnext - pos);
      // std::cout << "bcsub:" << bcsub << " b:" << b << std::endl;
      int bcrange = 0;
      try {
        bcrange = std::stoull(bcsub);
      } catch (...) {
        LOG(warning) << "problem in bcmask decoding H:" << posH << " posL:" << posL << " bcsub:" << bcsub;
        return 1;
      }
      if (b) {
        for (int bc = bccur; bc < bccur + bcrange; bc++) {
          try {
            BCmask.set(bc, 1);
          } catch (...) {
            LOG(warning) << "BC mask decoding to big bc:" << bc;
          }
        }
      }
      bccur += bcrange;
      pos = posnext + 1;
      // std::cout << "bccur:" << bccur << std::endl;
    }
  } else {
    // list of integers
    for (int i = 2; i < tokens.size(); i++) {
      uint32_t bc;
      try {
        bc = std::stoull(tokens[i]);
      } catch (...) {
        LOG(info) << "mask syntax:" << i << ":" << tokens[i];
        continue;
      }
      BCmask.set(bc, 1);
    }
  }
  return 0;
}
void BCMask::printStream(std::ostream& stream) const
{
  stream << "CTP BC mask:" << name << ":" << mask; /// << std::endl;
  stream << " # of active BC:" << BCmask.count() << std::endl;
}
//
const std::set<std::string> CTPGenerator::Generators = {"bcd1m", "bcd2m", "bcd10", "bcd20", "rnd1m", "rnd2m", "rnd10", "rnd20"};
void CTPGenerator::printStream(std::ostream& stream) const
{
  stream << "CTP generator:" << name << " frequency:" << frequency << std::endl;
}
//
CTPInput::CTPInput(std::string& name, std::string& det, uint32_t index)
{
  this->name = name;
  inputMask = (1ull << (index - 1));
  detID = o2::detectors::DetID(det.c_str());
}
CTPInput::CTPInput(const char* name, const char* det, uint32_t index)
{
  this->name = std::string(name);
  inputMask = (1ull << (index - 1));
  detID = o2::detectors::DetID(det);
}
void CTPInput::setRun3DetName(std::string& run2Name)
{
  std::string run3Name = CTPInput::run2DetToRun3Det.at(run2Name);
  detID = o2::detectors::DetID(run3Name.c_str());
}
void CTPInput::printStream(std::ostream& stream) const
{
  stream << "CTP Input:" << name << " Detector:" << getInputDetName() << " Level:" << level << " Hardware mask:0x" << std::hex << inputMask << std::dec;
  stream << " index:" << getIndex() << std::endl;
}
//
std::uint64_t CTPDescriptor::getInputsMask() const
{
  uint64_t mask = 0;
  for (const auto& inp : inputs) {
    mask |= inp->inputMask;
  }
  return mask;
}
void CTPDescriptor::printStream(std::ostream& stream) const
{
  stream << "CTP Descriptor:" << name << " Inputs:";
  for (const auto& inp : inputs) {
    stream << inp->name << " ";
  }
  stream << std::endl;
}
//
void CTPDetector::printStream(std::ostream& stream) const
{
  stream << "CTP Detector:" << getName() << " HBaccepted:" << HBaccepted;
  stream << " Mode:" << mode << " FErst:" << ferst << std::endl;
}

void CTPCluster::printStream(std::ostream& stream) const
{
  stream << "CTP Cluster:" << name << " " << getClusterDetNames();
  stream << " det mask:0b" << std::hex << maskCluster << " " << std::dec;
  stream << " clust index:" << hwMask;
  stream << std::endl;
}
void CTPClass::printStream(std::ostream& stream) const
{
  stream << "CTP Class:" << name << " Hardware mask:" << classMask << " Cluster index:" << clusterIndex << " Desc index:" << descriptorIndex;
  stream << " Downscale:" << downScale;
  stream << " BCM:";
  for (const auto& bcm : BCClassMask) {
    stream << bcm->name << " ";
  }
  if (descriptor != nullptr) {
    stream << " Descriptor:" << descriptor->name;
  }
  if (cluster != nullptr) {
    stream << " Cluster:" << cluster->name;
  }
  stream << std::endl;
}
/// CTP configuration
/// Assuming Run2 format + LTG
int CTPConfiguration::addInput(std::string& inp, int clsindex, std::map<int, std::vector<int>>& descInputsIndex)
{
  LOG(info) << "adding input:" << inp;
  CTPInput ctpinp;
  std::string sinp = inp;
  ;
  if (inp[0] == '~') {
    sinp = inp.substr(1, inp.size() - 1);
    ctpinp.neg = 0;
  }
  if (inp[0] == 'b') { // BC downscale
    ctpinp.level = "b";
    ctpinp.name = inp;
  } else if (inp[0] == 'r') { // randpm gen
    ctpinp.level = "r";
    ctpinp.name = inp;
  } else if (isNumber(sinp)) { // inputs as number
    int index = std::stoi(sinp);
    ctpinp.name = CTPInputsConfiguration::getInputNameFromIndex100(index);
    ctpinp.inputMask = 1ull << (index - 1);
    ctpinp.level = ctpinp.name[0];
    if (ctpinp.neg == 0) {
      ctpinp.name = "~" + ctpinp.name;
    }
  } else { // input as string or error
    ctpinp.name = inp;
    int index = CTPInputsConfiguration::getInputIndexFromName(inp);
    ctpinp.level = sinp[0];
    ctpinp.inputMask = 1ull << (index - 1);
  }
  // add to desc
  // check if already there
  for (int i = 0; i < mInputs.size(); i++) {
    if (mInputs[i].name == ctpinp.name) {
      LOG(info) << "input found at:" << i;
      descInputsIndex[clsindex].push_back(i);
      return 0;
    }
  }
  mInputs.push_back(ctpinp);
  descInputsIndex[clsindex].push_back(mInputs.size() - 1);
  LOG(info) << "input inderted at:" << mInputs.size() - 1;
  return 0;
}
int CTPConfiguration::loadConfigurationRun3(const std::string& ctpconfiguration)
{
  LOG(info) << "Loading CTP configuration.";
  std::map<int, std::vector<int>> clsDescIndex;
  CTPInputsConfiguration::initDefaultInputConfig();
  mConfigString = ctpconfiguration;
  std::istringstream iss(ctpconfiguration);
  int ret = 0;
  int level = START;
  std::string line;
  int ver = 0;
  int iline = 0;
  while (std::getline(iss, line)) {
    o2::utils::Str::trim(line);
    if (line.size() == 0) {
      continue;
    }
    if (line.at(0) == '#') {
      continue;
    }
    if (iline == 0) {
      if (line.find("ver") != std::string::npos) {
        ver = 1;
        LOG(info) << "CTP Config vesrion:" << line;
      } else {
        LOG(info) << "CTP Config version: 0";
      }
    }
    iline++;
    if (ver == 0) {
      ret = processConfigurationLineRun3(line, level, clsDescIndex);
    } else {
      ret = processConfigurationLineRun3v2(line, level, clsDescIndex);
    }
    if (ret) {
      return ret;
    }
  }
  if (ver == 0) {
    for (auto& cls : mCTPClasses) {
      cls.cluster = &mClusters[cls.clusterIndex];
      if (cls.descriptorIndex != 0xff) {
        cls.descriptor = &mDescriptors[cls.descriptorIndex];
        if (cls.getIndex() != 0xff) {
          for (auto const& inp : clsDescIndex[cls.getIndex()]) {
            mDescriptors.at(cls.descriptorIndex).inputs.push_back(&mInputs.at(inp));
          }
        }
      }
    }
    createInputsInDecriptorsFromNames();
  } else {
    for (auto& cls : mCTPClasses) {
      cls.cluster = &mClusters[cls.clusterIndex];
    }
  }
  return ret;
}
int CTPConfiguration::processConfigurationLineRun3(std::string& line, int& level, std::map<int, std::vector<int>>& descInputsIndex)
{
  LOG(info) << "Processing line";
  LOG(info) << "line:" << line << " lev:" << level;
  //
  std::vector<std::string> tokens = o2::utils::Str::tokenize(line, ' ');
  size_t ntokens = tokens.size();
  if (ntokens == 0) {
    LOG(warning) << "# of tokens zero in line:" << line;
    return 0;
  }
  size_t first;
  if (((first = line.find("run")) != std::string::npos) && (level == START)) {
    level = RUN;
  } else if ((line.find("inp") != std::string::npos) && ((level == RUN) || (level == INPUTS))) {
    level = INPUTS;
  } else if (((first = line.find("bcm")) != std::string::npos) && ((level == RUN) || (level == INPUTS) || (level == MASKS))) {
    level = MASKS;
  } else if (CTPGenerator::Generators.count(tokens[0]) && ((level == RUN) || (level == INPUTS) || (level == MASKS) || (level == GENS))) {
    level = GENS;
  } else if ((first = line.find("LTG")) != std::string::npos) {
    level = LTG;
  } else if ((first = line.find("cluster")) != std::string::npos) {
    level = CLUSTER;
  } else {
    bool knownlevels = ((level == LTGitems) || (level == CLASS));
    if (knownlevels == false) {
      level = UNKNOWN;
    }
  }
  LOG(info) << "Level:" << level;
  switch (level) {
    case RUN: {
      try {
        mRunNumber = std::stoul(tokens[1]);
      } catch (...) {
        LOG(error) << "RUN:" << tokens[1] << std::endl;
      }
      level = RUN;
      break;
    }
    case INPUTS: {
      level = INPUTS;
      if (tokens.size() < 3) {
        LOG(error) << "Wrong input line:" << line;
        break;
      }
      CTPInput ctpinp;
      ctpinp.name = tokens[1];
      ctpinp.level = tokens[1][0];
      std::string run2Name{tokens[1][1]};
      ctpinp.setRun3DetName(run2Name);
      uint32_t index = std::stoul(tokens[2]);
      ctpinp.inputMask = (1ull << (index - 1));
      mInputs.push_back(ctpinp);
      LOG(info) << "Input:" << ctpinp.name << " index:" << index;
      break;
    }
    case MASKS: {
      BCMask bcmask;
      if (tokens.size() < 3) {
        LOG(error) << "Wrong bc mask:" << line;
        break;
      }
      bcmask.name = tokens[1];
      bool coded = tokens[2].find("L") != std::string::npos;
      coded |= tokens[2].find("H") != std::string::npos;
      // std::cout << "coded:" << coded << std::endl;
      if (coded) {
        // jusko notation
      } else {
        // list of integers
        for (int i = 2; i < ntokens; i++) {
          uint32_t bc;
          try {
            bc = std::stoull(tokens[i]);
          } catch (...) {
            LOG(info) << "mask syntax:" << i << ":" << tokens[i];
            continue;
          }
          bcmask.BCmask.set(bc, 1);
        }
      }
      mBCMasks.push_back(bcmask);
      LOG(info) << "BC mask added:" << bcmask.name;
      break;
    }
    case GENS: {
      CTPGenerator gen;
      gen.name = tokens[0];
      gen.frequency = tokens[1];
      LOG(info) << "Gen added:" << line;
      break;
    }
    case LTG: {
      CTPDetector ctpdet;
      std::string detname = tokens[1];
      capitaliseString(detname);
      o2::detectors::DetID det(detname.c_str());
      if (isDetector(det)) {
        ctpdet.detID = det.getID();
        LOG(info) << "Detector found:" << det.getID() << " " << detname;
      } else {
        LOG(info) << "Unknown detectors:" << line;
      }
      mDetectors.push_back(ctpdet);
      level = LTGitems;
      break;
    }
    case LTGitems: {
      if (ntokens == 1) {
        mDetectors.back().mode = tokens[0];
      }
      LOG(info) << "LTGitem:" << line;
      break;
    }
    case CLUSTER: {
      CTPCluster cluster;
      try {
        cluster.hwMask = std::stoull(tokens[0]);
      } catch (...) {
        LOG(info) << "Cluster syntax error:" << line;
        return level;
      }
      LOG(info) << "Cluster:" << line;
      cluster.name = tokens[2];
      o2::detectors::DetID::mask_t mask;
      for (int item = 3; item < ntokens; item++) {
        std::string detname = tokens[item];
        capitaliseString(detname);
        // LOG(info) << "Detector:" << detname;
        o2::detectors::DetID det(detname.c_str());
        isDetector(det);
        mask |= det.getMask();
      }
      cluster.maskCluster = mask;
      mClusters.push_back(cluster);
      level = CLASS;
      // LOG(info) << "Cluster done:" << cluster.name << std::endl;
      break;
    }
    case CLASS: {
      // add to the last cluster
      if (tokens.size() < 3) {
        LOG(error) << "CTPClass: less than 3 items in class:" << line;
        break;
      }
      uint64_t index;
      try {
        index = std::stoull(tokens[0]);
      } catch (...) {
        LOG(info) << "Class syntax error:" << line;
        return level;
      }
      LOG(info) << "Class:" << line;
      CTPClass cls;
      cls.classMask = 1ull << index;
      cls.name = tokens[1];
      cls.clusterIndex = mClusters.size() - 1;
      CTPDescriptor desc;
      desc.name = "d" + cls.name;
      // LOG(info) << "point:" << cls.cluster << " " << &mClusters.front();
      for (int i = 2; i < tokens.size(); i++) {
        std::string token = tokens[i];
        bool isGenerator = 0;
        for (auto const& gen : CTPGenerator::Generators) {
          if (token.find(gen) != std::string::npos) {
            isGenerator = 1;
            break;
          }
        }
        if (isGenerator) {
          addInput(token, index, descInputsIndex);
          LOG(info) << "Class generator found:" << desc.name;
        } else if (token.find("~") != std::string::npos) { // inverted input
          // std::cout << "Inverted input" << std::endl;
          addInput(token, index, descInputsIndex);
        } else if (isNumber(token)) { // normal input as number
          // std::cout << "Normal input" << std::endl;
          addInput(token, index, descInputsIndex);
          // LOG(info) << "Class input descriptor:" << mDescriptors[mDescriptors.size() - 1].name;
        } else if (token.find("0x") != std::string::npos) { // downscale
          // std::cout << "Downscale" << std::endl;
          cls.downScale = std::stoul(token, nullptr, 16);
        } else if (token.find("bcm") != std::string::npos) { // bcmask
          // std::cout << "Mask" << std::endl;
          int i = 0;
          for (auto const& bcm : mBCMasks) {
            if (bcm.name == token) {
              cls.BCClassMask.push_back(&bcm);
              LOG(info) << "Class BCMask found:" << token;
              break;
            }
            i++;
          }
          if (i == mBCMasks.size()) {
            LOG(error) << "Class BCMask NOT found:" << token << " assuming input";
          }
        } else { // input as string or descriptor
          addInput(token, index, descInputsIndex);
        }
      }
      mDescriptors.push_back(desc);
      cls.descriptorIndex = mDescriptors.size() - 1;
      //
      mCTPClasses.push_back(cls);
      break;
    }
    default: {
      LOG(info) << "unknown line:" << line;
    }
  }
  return 0;
}
int CTPConfiguration::processConfigurationLineRun3v2(std::string& line, int& level, std::map<int, std::vector<int>>& descInputsIndex)
{
  LOG(info) << "Processing line";
  LOG(info) << "line:" << line << " lev:" << level;
  //
  std::vector<std::string> tokens = o2::utils::Str::tokenize(line, ' ');
  size_t ntokens = tokens.size();
  if (ntokens == 0) {
    LOG(warning) << "# of tokens zero in line:" << line;
    return 0;
  }
  size_t first;
  if (((first = line.find("ver")) != std::string::npos) && (level == START)) {
    mVersion = line;
    // std::cout << "debug:" << mVersion << std::endl;
    level = VERSION;
    return 0;
  } else if (((first = line.find("run")) != std::string::npos) && (level == VERSION)) {
    level = RUN;
  } else if ((line.find("INPUTS") != std::string::npos) && (level == RUN)) {
    level = INPUTS;
    return 0;
  } else if ((line.find("inp") != std::string::npos) && (level == INPUTS)) {
    level = INPUTS;
  } else if ((line.find("BCMASKS") != std::string::npos) && ((level == INPUTS) || (level == RUN))) {
    level = MASKS;
    return 0;
  } else if (((first = line.find("bcm")) != std::string::npos) && (level == MASKS)) {
    level = MASKS;
  } else if (line.find("GENS") != std::string::npos) {
    level = GENS;
    return 0;
  } else if (CTPGenerator::Generators.count(tokens[0]) && (level == GENS)) {
    level = GENS;
  } else if (line.find("DESCRIPTORS") != std::string::npos) {
    level = DESCRIPTORS;
    return 0;
  } else if ((tokens[0][0] == 'D') && (level == DESCRIPTORS)) {
    level = DESCRIPTORS;
  } else if ((first = line.find("LTG")) != std::string::npos) {
    level = LTG;
  } else if ((first = line.find("cluster")) != std::string::npos) {
    level = CLUSTER;
  } else {
    bool knownlevels = ((level == LTGitems) || (level == CLASS));
    if (knownlevels == false) {
      level = UNKNOWN;
    }
  }
  LOG(info) << "Level:" << level;
  switch (level) {
    case VERSION: {
      break;
    }
    case RUN: {
      try {
        mRunNumber = std::stoul(tokens[1]);
      } catch (...) {
        LOG(error) << "RUN line:" << line;
      }
      level = RUN;
      break;
    }
    case INPUTS: {
      level = INPUTS;
      if (tokens.size() != 3) {
        LOG(error) << "Wrong input line:" << line;
        return 1;
      }
      CTPInput ctpinp;
      ctpinp.name = tokens[1];
      ctpinp.level = tokens[1][0];
      std::string run2Name{tokens[1][1]};
      ctpinp.setRun3DetName(run2Name);
      uint32_t index = std::stoul(tokens[2]);
      ctpinp.inputMask = (1ull << (index - 1));
      mInputs.push_back(ctpinp);
      LOG(info) << "Input:" << ctpinp.name << " index:" << index;
      break;
    }
    case MASKS: {
      BCMask bcmask;
      if (tokens.size() < 3) {
        LOG(error) << "Wrong bc mask:" << line;
        break;
      }
      bcmask.setBCmask(tokens);
      mBCMasks.push_back(bcmask);
      LOG(info) << "BC mask added:" << bcmask.name;
      break;
    }
    case GENS: {
      CTPGenerator gen;
      gen.name = tokens[0];
      gen.frequency = tokens[1];
      mGenerators.push_back(gen);
      LOG(info) << "Gen added:" << line;
      break;
    }
    case DESCRIPTORS: {
      if ((tokens.size() < 2) && (line.find("DTRUE") == std::string::npos)) {
        LOG(warning) << "Dsecriptor:" << line;
        break;
      }
      CTPDescriptor desc;
      desc.name = tokens[0];
      for (int i = 1; i < tokens.size(); i++) {
        const CTPInput* inp = isInputInConfig(tokens[i]);
        if (inp != nullptr) {
          desc.inputs.push_back(inp);
        }
      }
      mDescriptors.push_back(desc);
      break;
    }
    case LTG: {
      CTPDetector ctpdet;
      std::string detname = tokens[1];
      capitaliseString(detname);
      o2::detectors::DetID det(detname.c_str());
      if (isDetector(det)) {
        ctpdet.detID = det.getID();
        LOG(info) << "Detector found:" << det.getID() << " " << detname;
      } else {
        LOG(info) << "Unknown detectors:" << line;
      }
      mDetectors.push_back(ctpdet);
      level = LTGitems;
      break;
    }
    case LTGitems: {
      if (ntokens == 1) {
        mDetectors.back().mode = tokens[0];
      }
      LOG(info) << "LTGitem:" << line;
      break;
    }
    case CLUSTER: {
      CTPCluster cluster;
      try {
        cluster.hwMask = std::stoull(tokens[0]);
      } catch (...) {
        LOG(info) << "Cluster syntax error:" << line;
        return level;
      }
      LOG(info) << "Cluster:" << line;
      cluster.name = tokens[2];
      o2::detectors::DetID::mask_t mask;
      for (int item = 3; item < ntokens; item++) {
        std::string detname = tokens[item];
        capitaliseString(detname);
        // LOG(info) << "Detector:" << detname;
        o2::detectors::DetID det(detname.c_str());
        isDetector(det);
        mask |= det.getMask();
      }
      cluster.maskCluster = mask;
      mClusters.push_back(cluster);
      level = CLASS;
      // LOG(info) << "Cluster done:" << cluster.name << std::endl;
      break;
    }
    case CLASS: {
      // add to the last cluster
      if (tokens.size() < 6) {
        LOG(error) << "CTPClass items < 6" << line;
        break;
      }
      uint64_t index;
      try {
        index = std::stoull(tokens[1]);
      } catch (...) {
        LOG(info) << "Class syntax error:" << line;
        return level;
      }
      LOG(info) << "Class:" << line;
      CTPClass cls;
      cls.classMask = 1ull << index;
      cls.name = tokens[0];
      // Descriptor
      std::string descname = tokens[2];
      int dindex;
      if (descname.find("DTRUE") != std::string::npos) {
        descname = "DTRUE";
      }
      const CTPDescriptor* desc = isDescriptorInConfig(descname, dindex);
      if (desc != nullptr) {
        cls.descriptor = desc;
        cls.descriptorIndex = dindex;
      }
      cls.clusterIndex = mClusters.size() - 1;
      // PF not member of class
      std::string bcmask = tokens[5];
      bcmask = bcmask.substr(1, bcmask.size() - 2);
      if (bcmask.size()) {
        const BCMask* bcm = isBCMaskInConfigP(bcmask);
        if (bcm != nullptr) {
          cls.BCClassMask.push_back(bcm);
        }
      }
      // Down scaling
      if (tokens.size() > 6) {
        cls.downScale = std::stoul(tokens[6], nullptr, 16);
      }
      mCTPClasses.push_back(cls);
      break;
    }
    default: {
      LOG(info) << "unknown line:" << line;
    }
  }
  return 0;
}
void CTPConfiguration::printStream(std::ostream& stream) const
{
  stream << "Configuration:" << mName << " Version:" << mVersion << std::endl;
  stream << "Run:" << mRunNumber << " cfg name:" << mName << std::endl;
  stream << "CTP BC  masks:" << std::endl;
  for (const auto& i : mBCMasks) {
    i.printStream(stream);
  }
  stream << "CTP inputs:" << mInputs.size() << std::endl;
  for (const auto& i : mInputs) {
    i.printStream(stream);
  }
  stream << "CTP generators:" << std::endl;
  for (const auto& i : mGenerators) {
    i.printStream(stream);
  }
  stream << "CTP descriptors:" << mDescriptors.size() << std::endl;
  for (const auto& i : mDescriptors) {
    i.printStream(stream);
  }
  stream << "CTP detectors:" << mDetectors.size() << std::endl;
  for (const auto& i : mDetectors) {
    i.printStream(stream);
  }
  stream << "CTP clusters:" << std::endl;
  for (const auto& i : mClusters) {
    i.printStream(stream);
  }
  stream << "CTP classes:" << std::endl;
  for (const auto& i : mCTPClasses) {
    i.printStream(stream);
  }
}
uint64_t CTPConfiguration::getInputMask(const std::string& name) const
{
  for (auto const& inp : mInputs) {
    if (inp.name == name) {
      return inp.inputMask;
    }
  }
  return 0;
}
int CTPConfiguration::getInputIndex(const std::string& name) const
{
  int index = 0xff;
  const CTPInput* inp = isInputInConfig(name);
  if (inp != nullptr) {
    index = inp->getIndex();
  }
  LOG(info) << "input:" << name << " index:" << index;
  return index;
}
bool CTPConfiguration::isMaskInInputs(const uint64_t& mask) const
{
  for (auto const& inp : mInputs) {
    if (inp.inputMask == mask) {
      return true;
    }
  }
  return false;
}
bool CTPConfiguration::isBCMaskInConfig(const std::string maskname) const
{
  for (auto& bcm : mBCMasks) {
    if (bcm.name == maskname) {
      return true;
    }
  }
  return false;
}
const BCMask* CTPConfiguration::isBCMaskInConfigP(const std::string maskname) const
{
  for (const auto& bcm : mBCMasks) {
    if (bcm.name == maskname) {
      LOG(info) << "isBCMaskInConfigP found:" << maskname;
      return &bcm;
    }
  }
  LOG(info) << "isBCMaskInConfigP NOT found:" << maskname;
  return nullptr;
}
const CTPInput* CTPConfiguration::isInputInConfig(const std::string inpname) const
{
  for (const auto& inp : mInputs) {
    if (inp.name == inpname) {
      LOG(info) << "isInputInConfig found:" << inpname;
      return &inp;
    }
  }
  LOG(info) << "isInputInConfig NOT found:" << inpname;
  return nullptr;
}
const CTPInput* CTPConfiguration::isInputInConfig(const int index) const
{
  for (const auto& inp : mInputs) {
    // std::cout << "isInputINConfig:" << inp.name << " " << inp.getIndex() << " " << index << std::endl;
    if (inp.getIndex() == index) {
      LOG(info) << "Found input:" << inp.name << " index:" << inp.getIndex();
      return &inp;
    }
  }
  return nullptr;
}
const CTPDescriptor* CTPConfiguration::isDescriptorInConfig(const std::string descname, int& index) const
{
  index = 0;
  for (const auto& desc : mDescriptors) {
    if (desc.name == descname) {
      LOG(info) << "isDescriptorInConfig found:" << descname;
      return &desc;
    }
    index++;
  }
  LOG(info) << "isDescriptorInConfig NOT found:" << descname;
  return nullptr;
}
void CTPConfiguration::createInputsInDecriptorsFromNames()
// using run3 conventions for inputs
{
  LOG(info) << "Creating Inputs";
  for (auto& des : mDescriptors) {
    if (CTPConfiguration::isNumber(des.name)) {
      // parse here if more inputs
      uint32_t index = std::stoul(des.name);
      if (index > 100) {
        index = index - 100;
      }
      // CTPInput* inp = const_cast<CTPInput*>(isInputInConfig(index));
      LOG(info) << "Desc index:" << index;
      const CTPInput* inp = isInputInConfig(index);
      if (inp) {
        des.inputs.push_back(inp);
      } else {
        LOG(warning) << "Descriptor not found:" << des.name;
      }
    } else {
      LOG(info) << "Input is not a number:" << des.name;
    }
  }
}
std::map<o2::detectors::DetID::ID, std::vector<CTPInput>> CTPConfiguration::getDet2InputMap()
{
  std::map<o2::detectors::DetID::ID, std::vector<CTPInput>> det2inp;
  for (auto const& inp : mInputs) {
    det2inp[inp.detID].push_back(inp);
  }
  return det2inp;
}
uint64_t CTPConfiguration::getTriggerClassMask() const
{
  uint64_t clsmask = 0;
  for (auto const& cls : mCTPClasses) {
    clsmask |= cls.classMask;
  }
  return clsmask;
}
std::vector<int> CTPConfiguration::getTriggerClassList() const
{
  uint64_t clsmask = getTriggerClassMask();
  std::vector<int> classlist;
  for (int i = 0; i < 64; i++) {
    if ((1ull << i) & clsmask) {
      classlist.push_back(i);
    }
  }
  return classlist;
}
std::vector<std::string> CTPConfiguration::getDetectorList() const
{
  std::vector<std::string> detlist;
  for (auto const& det : mDetectors) {
    std::string sdet(det.getName());
    detlist.push_back(sdet);
  }
  return detlist;
}
o2::detectors::DetID::mask_t CTPConfiguration::getDetectorMask() const
{
  o2::detectors::DetID::mask_t mask = 0;
  for (auto const& det : mDetectors) {
    mask |= det.getMask();
  }
  return mask;
}
// This is special case of general approach:
// classmask = fin(inputmask)g
uint64_t CTPConfiguration::getClassMaskForInputMask(uint64_t inputMask) const
{
  uint64_t clsmask = 0;
  for (auto const& cls : mCTPClasses) {
    if (cls.descriptor) {
      // std::cout << cls.name << std::hex << " " << cls.descriptor->getInputsMask() << " " << inputMask << std::endl;
      if (cls.descriptor->getInputsMask() & inputMask) {
        clsmask += cls.classMask;
        // std::cout << " clsmask:" << clsmask << std::endl;
      }
    }
  }
  return clsmask;
}
int CTPConfiguration::assignDescriptors()
{
  for (auto& cls : mCTPClasses) {
    cls.descriptor = &mDescriptors[cls.descriptorIndex];
  }
  return 0;
}
int CTPConfiguration::checkConfigConsistency() const
{
  LOG(info) << "Checking consistency run:" << mRunNumber;
  int ret = 0;
  // All inputs used ?
  // std::map<const CTPInput*, int> inputs;
  std::map<std::string, int> inputs;
  for (auto const& inp : mInputs) {
    inputs[inp.name] = 0;
  }
  // Are all descriptors used
  // std::map<const CTPDescriptor*, int> descs;
  std::map<std::string, int> descs;
  for (auto const& desc : mDescriptors) {
    descs[desc.name] = 0;
    // std::cout << "1 " << &desc << std::endl;
    for (auto const inp : desc.inputs) {
      inputs[inp->name] += 1;
    }
  }
  std::cout << "desc1:" << descs.size() << std::endl;
  //
  for (const auto& cls : mCTPClasses) {
    if (cls.classMask == 0) {
      std::cout << "ERROR class:" << cls.name << " NO CLASS MASK" << std::endl;
      ret++;
    }
    if (cls.cluster == nullptr) {
      std::cout << "ERROR class:" << cls.name << " NO CLUSTER" << std::endl;
      ret++;
    }
    if (cls.clusterIndex == 0xff) {
      std::cout << "ERROR class:" << cls.name << " NO CLUSTER INDEX" << std::endl;
      ret++;
    }
    if (cls.descriptor == nullptr) {
      std::cout << "ERROR class:" << cls.name << " NO DESCRIPTOR" << std::endl;
      ret++;
    } else {
      descs[cls.descriptor->name] += 1;
      // std::cout << "2 " << cls.descriptor << std::endl;
    }
    if (cls.descriptorIndex == 0xff) {
      std::cout << "ERROR class:" << cls.name << " NO DESCRIPTOR INDEX" << std::endl;
      ret++;
    } else {
      // std::cout << "3 " << &mDescriptors[cls.descriptorIndex] << std::endl;
    }
  }
  int iw = 0;
  for (auto const& inp : inputs) {
    if (inp.second == 0) {
      iw++;
      std::cout << "WARNING inputs:";
    }
    std::cout << inp.first << " " << inp.second << std::endl;
  }
  std::cout << "Descriptors check:" << descs.size() << std::endl;
  for (auto const& desc : descs) {
    if (desc.second == 0) {
      iw++;
      std::cout << "WARNING descriptors:";
    }
    // std::cout << (desc.first)->name << " " << desc.second << std::endl;
    std::cout << (desc.first) << " " << desc.second << std::endl;
  }
  std::cout << "CTP Config consistency checked. WARNINGS:" << iw << " ERRORS:" << ret << std::endl;
  return ret;
}
//
int CTPInputsConfiguration::createInputsConfigFromFile(std::string& filename)
{
  int ret = 0;
  std::ifstream inpcfg(filename);
  if (inpcfg.is_open()) {
    std::string line;
    while (std::getline(inpcfg, line)) {
      o2::utils::Str::trim(line);
      if (line.size() == 0) {
        continue;
      }
      if (line[0] == '#') {
        continue;
      }
      std::vector<std::string> tokens = o2::utils::Str::tokenize(line, ' ');
      size_t ntokens = tokens.size();
      if (ntokens < 6) {
        LOG(warning) << "# of tokens < 6 in line:" << ntokens << ":" << line;
        ret++;
      } else {
        CTPInput inp;
        uint32_t index = 0;
        try {
          index = std::stoi(tokens[0]);
        } catch (...) {
          LOG(warning) << line;
          ret++;
          continue;
        }
        std::string det = tokens[1];
        CTPConfiguration::capitaliseString(det);
        std::string name = tokens[2];
        CTPInputs.push_back(CTPInput(name, det, index));
      }
    }
  } else {
    LOG(info) << "Can not open file:" << filename;
    ret++;
  }
  return ret;
}
///
/// CTP inputs config
/// Only default used.
///
CTPInputsConfiguration CTPInputsConfiguration::defaultInputConfig;
void CTPInputsConfiguration::printStream(std::ostream& stream) const
{
  for (auto const& input : CTPInputs) {
    input.printStream(stream);
  }
}
const std::vector<CTPInput> CTPInputsConfiguration::CTPInputsDefault =
  {
    CTPInput("MT0A", "FT0", 1), CTPInput("MT0C", "FT0", 2), CTPInput("MTVX", "FT0", 3), CTPInput("MTSC", "FT0", 4), CTPInput("MTCE", "FT0", 5),
    CTPInput("MVBA", "FV0", 6), CTPInput("MVOR", "FV0", 7), CTPInput("MVIR", "FV0", 8), CTPInput("MVNC", "FV0", 9), CTPInput("MVCH", "FV0", 10),
    CTPInput("0UCE", "FDD", 13), CTPInput("0USC", "FDD", 15), CTPInput("0UVX", "FDD", 16), CTPInput("0U0C", "FDD", 17), CTPInput("0U0A", "FDD", 18),
    CTPInput("0DMC", "EMC", 14), CTPInput("0DJ1", "EMC", 41), CTPInput("0DG1", "EMC", 42), CTPInput("0DJ2", "EMC", 43), CTPInput("0DG2", "EMC", 44),
    CTPInput("0EMC", "EMC", 21), CTPInput("0EJ1", "EMC", 37), CTPInput("0EG1", "EMC", 38), CTPInput("0EJ2", "EMC", 39), CTPInput("0EG2", "EMC", 40),
    CTPInput("0PH0", "PHS", 22), CTPInput("1PHL", "PHS", 27), CTPInput("1PHH", "PHS", 28), CTPInput("1PHL", "PHM", 29),
    CTPInput("1ZED", "ZDC", 25), CTPInput("1ZNC", "ZDC", 26)};
void CTPInputsConfiguration::initDefaultInputConfig()
{
  defaultInputConfig.CTPInputs = CTPInputsConfiguration::CTPInputsDefault;
}
/// Return input name from default inputs configuration.
/// Take into account convention that L0 inputs has (index+100) in the first version of CTP config file (*.rcfg)
std::string CTPInputsConfiguration::getInputNameFromIndex100(int index)
{
  int indexcor = index;
  if (index > 100) {
    indexcor = index - 100;
  }
  for (auto& inp : defaultInputConfig.CTPInputs) {
    if (inp.getIndex() == indexcor) {
      std::string name = inp.name;
      if (index > 100) {
        name[0] = '0';
      }
      return name;
    }
  }
  LOG(info) << "Input with index:" << index << " not in deafult input config";
  return "";
}
/// Return input name from default inputs configuration.
/// Index has to be in range [1::48]
std::string CTPInputsConfiguration::getInputNameFromIndex(int index)
{
  if (index > o2::ctp::CTP_NINPUTS) {
    LOG(warn) << "getInputNameFRomIndex: index too big:" << index;
    return "none";
  }
  for (auto& inp : o2::ctp::CTPInputsConfiguration::CTPInputsDefault) {
    if (inp.getIndex() == index) {
      std::string name = inp.name;
      return name;
    }
  }
  LOG(info) << "Input with index:" << index << " not in deafult input config";
  return "none";
}
int CTPInputsConfiguration::getInputIndexFromName(std::string& name)
{
  std::string namecorr = name;
  if ((name[0] == '0') || (name[0] == 'M') || (name[0] == '1')) {
    namecorr.substr(1, namecorr.size() - 1);
  } else {
    LOG(warn) << "Input name without level:" << name;
  }
  for (auto& inp : o2::ctp::CTPInputsConfiguration::CTPInputsDefault) {
    if (inp.name.find(namecorr) != std::string::npos) {
      return inp.getIndex();
    }
  }
  LOG(warn) << "Input with name:" << name << " not in default input config";
  return 0xff;
}
