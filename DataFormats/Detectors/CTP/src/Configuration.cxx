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
#include <iostream>
#include <sstream>
#include <regex>
#include "CommonUtils/StringUtils.h"
#include "FairLogger.h"

using namespace o2::ctp;
//
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
//
void BCMask::printStream(std::ostream& stream) const
{
  stream << "CTP BC mask:" << name << std::endl;
  /// <<  ":" << BCmask << std::endl;
}
//
const std::set<std::string> CTPGenerator::Generators = {"bcd1m", "bcd2m", "bcd10", "bcd20", "rnd1m", "rnd2m", "rnd10", "rnd20"};
void CTPGenerator::printStream(std::ostream& stream) const
{
  stream << "CTP generator:" << name << " frequency:" << frequency << std::endl;
}
//
void CTPInput::printStream(std::ostream& stream) const
{
  stream << "CTP Input:" << name << " Detector:" << getInputDetName() << " Level:" << level << " Hardware mask:0x" << std::hex << inputMask << std::dec << std::endl;
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
  stream << " mask:0b" << std::hex << maskCluster << " " << std::dec;
  stream << std::endl;
}
//
void CTPClass::printStream(std::ostream& stream) const
{
  stream << "CTP Class:" << name << " Hardware mask:" << classMask;
  if(descriptor != nullptr) {
    stream << " Descriptor:" << descriptor->name;
  }
  if(cluster != nullptr) {
    stream << " Cluster:" << cluster->name;
  }
  stream << std::endl;
}
/// CTP configuration
/// Assuming Run2 format + LTG
int CTPConfiguration::loadConfigurationRun3(const std::string& ctpconfiguration)
{
  LOG(info) << "Loading CTP configuration.";
  std::istringstream iss(ctpconfiguration);
  int ret = 0;
  int level = MASKS;
  std::string line;
  while (std::getline(iss, line)) {
    o2::utils::Str::trim(line);
    if ((ret = processConfigurationLineRun3(line, level)) != 0) {
      return ret;
    }
  }
  return ret;
  return 0;
}
int CTPConfiguration::loadConfiguration(const std::string& ctpconfiguration)
{
  LOG(info) << "Loading CTP configuration.";
  std::istringstream iss(ctpconfiguration);
  int ret = 0;
  int level = 0;
  std::string line;
  while (std::getline(iss, line)) {
    o2::utils::Str::trim(line);
    if ((ret = processConfigurationLine(line, level)) != 0) {
      return ret;
    }
  }
  return ret;
}
int CTPConfiguration::processConfigurationLineRun3(std::string& line, int& level)
{
  // LOG(debug) << "line:" << line;
  if (line.size() == 0) {
    return 0;
  }
  if (line.at(0) == '#') {
    return 0;
  }
  //
  std::vector<std::string> tokens = o2::utils::Str::tokenize(line, ' ');
  size_t ntokens = tokens.size();
  if (ntokens == 0) {
    LOG(warning) << "# of tokens zero in line:" << line;
    return 0;
  }
  size_t first;
  if (CTPGenerator::Generators.count(tokens[0])) {
    if (level != CLASS) {
      level = GENS;
    }
    level = GENS;
  } else if ((first = line.find("bcm")) != std::string::npos) {
    if (level == MASKS) {
      level = MASKS;
    }
  } else if ((first = line.find("LTG")) != std::string::npos) {
    level = LTG;
  } else if ((first = line.find("cluster")) != std::string::npos) {
    level = CLUSTER;
  } else {
  }
  switch (level) {
    case MASKS: {
      BCMask bcmask;
      bcmask.name = tokens[1];
      std::regex LHdelimiters("[LH]");
      if (std::regex_search(tokens[2], LHdelimiters)) {
        // jusko notation
      } else {
        // list of integers
        for (int i = 2; i < ntokens; i++) {
          uint32_t bc;
          try {
            bc = std::stoull(tokens[i]);
          } catch (...) {
            LOG(info) << "mask syntax:" << tokens[i];
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
      // LOG(info) << "point:" << cls.cluster << " " << &mClusters.front();
      mCTPClasses.push_back(cls);
      break;
    }
    default: {
      LOG(info) << "unknown line:" << line;
    }
  }
  for (auto& cls : mCTPClasses) {
    cls.cluster = &mClusters[cls.clusterIndex];
  }
  return 0;
}
int CTPConfiguration::processConfigurationLine(std::string& line, int& level)
{
  if (line.size() == 0) {
    return 0;
  }
  if (line.at(0) == '#') {
    return 0;
  }
  size_t first;
  if ((first = line.find("PARTITION:")) != std::string::npos) {
    mName = o2::utils::Str::trim_copy(line.erase(first, 10));
    return 0;
  }
  if ((first = line.find("VERSION:")) != std::string::npos) {
    mVersion = o2::utils::Str::trim_copy(line.erase(first, 8));
    return 0;
  }
  if ((first = line.find("INPUTS:")) != std::string::npos) {
    level = 1;
    return 0;
  }
  if ((first = line.find("DESCRIPTORS:")) != std::string::npos) {
    level = 3;
    return 0;
  }
  if ((first = line.find("CLUSTERS:")) != std::string::npos) {
    level = 4;
    return 0;
  }
  if ((first = line.find("CLASSES:")) != std::string::npos) {
    level = 7;
    return 0;
  }
  /// Do parse levels
  std::vector<std::string> tokens = o2::utils::Str::tokenize(line, ' ');
  size_t ntokens = tokens.size();
  if (ntokens == 0) {
    return 0;
  }
  switch (level) {
    case 1:
      /// INPUTS: name det level indexCTP<0:45>
      {
        if (ntokens != 4) {
          LOG(fatal) << "INPUTS syntax error in, wrong number of items, expected 4:" << line;
          return level;
        }
        CTPInput inp;
        inp.name = tokens[0];
        std::string detName = tokens[1];
        inp.level = tokens[2];
        o2::detectors::DetID det(detName.c_str());
        isDetector(det);
        inp.detID = det.getID();
        //std::cout << "id:" << det.getID() << " ndets:" << det.getNDetectors() << std::endl;
        try {
          inp.inputMask = std::stoull(tokens[3], nullptr, 0);
        } catch (...) {
          LOG(fatal) << "INPUTS syntax error in mask:" << line;
          return level;
        }
        mInputs.push_back(inp);
        break;
      }
    case 3:
      /// Descriptors: name input1 , input2, ...
      {
        CTPDescriptor desc;
        desc.name = tokens[0];
        tokens.erase(tokens.begin());
        for (auto& item : tokens) {
          //CTPInput *inp = const_cast<CTPInput*> (isInputInConfig(item));
          CTPInput* inp = isInputInConfig(item);
          if (inp == nullptr) {
            LOG(fatal) << "DESCRIPTOR: input not in INPUTS:" << item << " LINE:" << line;
          } else {
            desc.inputs.push_back(inp);
          }
        }
        // Create inputs and mask
        mDescriptors.push_back(desc);
        break;
      }
    case 4:
      /// Clusters: name det1 det2 ... det N
      {
        CTPCluster cluster;
        cluster.name = tokens[0];
        tokens.erase(tokens.begin());
        o2::detectors::DetID::mask_t mask;
        for (auto& item : tokens) {
          o2::detectors::DetID det(item.c_str());
          isDetector(det);
          mask |= det.getMask();
        }
        cluster.maskCluster = mask;
        mClusters.push_back(cluster);
        break;
      }
    case 7:
      /// CLASSES: name mask descriptor cluster LM L0 L1
      {
        CTPClass cls;
        if (ntokens != 4) {
          LOG(fatal) << "CLASSES syntax error, wrong number of items, expected 4:" << line;
          return level;
        }
        cls.name = tokens[0];
        try {
          cls.classMask = std::stoull(tokens[1]);
        } catch (...) {
          LOG(fatal) << "CLASSES syntax error in mask:" << line;
          return level;
        }
        std::string token = tokens[2];
        auto it = std::find_if(mDescriptors.begin(), mDescriptors.end(), [&token](const CTPDescriptor& obj) { return obj.name == token; });
        if (it != mDescriptors.end()) {
          cls.descriptor = &*it;
        } else {
          ///Internal error
          LOG(fatal) << "CLASSES syntax error, descriptor not found:" << token;
        }
        ///
        token = tokens[3];
        auto it2 = std::find_if(mClusters.begin(), mClusters.end(), [&token](const CTPCluster& obj) { return obj.name == token; });
        if (it2 != mClusters.end()) {
          cls.cluster = &*it2;
        } else {
          ///Internal error
          LOG(fatal) << "CLASSES syntax error, cluster not found:" << token;
        }
        mCTPClasses.push_back(cls);
        break;
      }
    default: {
      LOG(fatal) << "CTP Config parser Unknown level:" << level;
    }
  }
  return 0;
}
void CTPConfiguration::addBCMask(const BCMask& bcmask)
{
  mBCMasks.push_back(bcmask);
}
void CTPConfiguration::addCTPInput(const CTPInput& input)
{
  mInputs.push_back(input);
}
void CTPConfiguration::addCTPDescriptor(const CTPDescriptor& descriptor)
{
  mDescriptors.push_back(descriptor);
}
void CTPConfiguration::addCTPDetector(const CTPDetector& detector)
{
  mDetectors.push_back(detector);
}
void CTPConfiguration::addCTPCluster(const CTPCluster& cluster)
{
  mClusters.push_back(cluster);
}
void CTPConfiguration::addCTPClass(const CTPClass& ctpclass)
{
  mCTPClasses.push_back(ctpclass);
}
void CTPConfiguration::printStream(std::ostream& stream) const
{
  stream << "Configuration:" << mName << "\n Version:" << mVersion << std::endl;
  stream << "CTP BC  masks:" << std::endl;
  for (const auto& i : mBCMasks) {
    i.printStream(stream);
  }
  stream << "CTP inputs:" << std::endl;
  for (const auto& i : mInputs) {
    i.printStream(stream);
  }
  stream << "CTP descriptors:" << std::endl;
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
uint64_t CTPConfiguration::getInputMask(const std::string& name)
{
  for (auto const& inp : mInputs) {
    if (inp.name == name) {
      return inp.inputMask;
    }
  }
  return 0;
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
CTPInput* CTPConfiguration::isInputInConfig(const std::string inpname)
{
  for (auto& inp : mInputs) {
    if (inp.name == inpname) {
      return &inp;
    }
  }
  return nullptr;
}
uint64_t CTPConfiguration::getDecrtiptorInputsMask(const std::string& name) const
{
  for (auto const& desc : mDescriptors) {
    if (desc.name == name) {
      return desc.getInputsMask();
    }
  }
  return 0xffffffff;
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
