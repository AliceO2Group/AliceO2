// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Configuration.cxx
/// \author Roman Lietava

#include "DataFormatsCTP/Configuration.h"
#include <iostream>
#include <sstream>
#include "CommonUtils/StringUtils.h"

using namespace o2::ctp;

void BCMask::printStream(std::ostream& stream) const
{
  stream << "CTP BC mask:" << name << std::endl;
  /// <<  ":" << BCmask << std::endl;
}
void CTPInput::printStream(std::ostream& stream) const
{
  stream << "CTP Input:" << name << " Detector:" << detName << " Level:" << level << " Hardware mask:0x" << std::hex << inputMask << std::dec << std::endl;
}
void CTPDescriptor::printStream(std::ostream& stream) const
{
  stream << "CTP Descriptor:" << name << " Inputs:";
  for(const auto &inp : inputNames){ stream << inp << " "; }
  stream << std::endl;
}
void CTPDetector::printStream(std::ostream& stream) const
{
  o2::detectors::DetID det(detID);
  stream << "CTP Detector:" << det.getName() << " HBaccepted:" << HBaccepted << std::endl;
}
void CTPCluster::printStream(std::ostream& stream) const
{
  stream << "CTP Cluster:" << name << " mask:0b" << std::hex << detectorsMask << " "<< std::dec;
  for(const auto &det : detectors){ stream << det.getName() << " ";}
  stream << std::endl;
}
void CTPClass::printStream(std::ostream& stream) const
{
  stream << "CTP Class:" << name << " Hardware mask:" << classMask << " Descriptor:" << descriptor->name << " Cluster:"<< cluster->name << std::endl;
}
/// CTP configuration
/// Assuming Run2 format + LTG
int CTPConfiguration::loadConfiguration(std::string& ctpconfiguration)
{
  std::cout << "Loading CTP configuration." << std::endl;  
  std::istringstream iss(ctpconfiguration);
  int ret = 0;
  int level=0;
  std::string line;
  while(std::getline(iss,line))
  {
    o2::utils::Str::trim(line);
    if( (ret = processConfigurationLine(line,level)) !=0 )
    {
        return ret;
    }
  }
  return ret;    
}
int CTPConfiguration::processConfigurationLine(std::string& line, int& level)
{
  if(line.size() == 0) return 0;
  if(line.at(0) == '#') return 0;
  size_t first;  
  if((first=line.find("PARTITION:")) != std::string::npos)
  {
    mName = o2::utils::Str::trim_copy(line.erase(first,10));
    return 0;
  }
  if((first=line.find("VERSION:")) != std::string::npos)
  {
    mVersion = o2::utils::Str::trim_copy(line.erase(first,8));
    return 0;
  }
  if((first=line.find("INPUTS:")) != std::string::npos)
  {
    level = 1;
    return 0;
  }
  if((first=line.find("DESCRIPTORS:")) != std::string::npos)
  {
    level = 3;
    return 0;
  }
  if((first=line.find("CLUSTERS:")) != std::string::npos)
  {
      level = 4;
      return 0;
  }
  if((first=line.find("CLASSES:")) != std::string::npos)
  {
    level = 7;
    return 0;
  }
  /// Do parse levels
  std::vector<std::string> tokens = o2::utils::Str::tokenize(line,' ');
  size_t ntokens=tokens.size();
  if(ntokens == 0) return 0;
  switch (level)
  {
    case 1:
    /// INPUTS: name det level indexCTP<0:45>
    {
        if(ntokens != 4)
        {
            std::cout << "Syntax error in INPUTS:" << line << std::endl;
            return level;
        }
        CTPInput inp;
        inp.name = tokens[0];
        inp.detName = tokens[1];
        inp.level = tokens[2];
        o2::detectors::DetID det(inp.detName.c_str());
        //std::cout << "id:" << det.getID() << " ndets:" << det.getNDetectors() << std::endl;
        if(det.getID() >= det.getNDetectors())
        {
            std::cout << "INPUT line:" << line << " Detector does not exist: " << inp.detName << std::endl;
            exit(1);
        }
        try
        {
            inp.inputMask = std::stoull(tokens[3],nullptr,0);
        }
        catch(...)
        {
            std::cout << "Syntax error in INPUTS in mask:" << line << std::endl;
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
        for(auto &item: tokens)
        {
            desc.inputNames.push_back(item);
        }
        /// Create inputs and mask
        mDescriptors.push_back(desc);
        break;
    }
    case 4:
    /// Clusters: name det1 det2 ... det N
    {
        CTPCluster cluster;
        cluster.name = tokens[0];
        tokens.erase(tokens.begin());
        o2::detectors::DetID::mask_t maskCluster;
        for(auto &item: tokens)
        {
            o2::detectors::DetID det(item.c_str());
            cluster.detectors.push_back(det);
            maskCluster |= det.getMask();
        }
        cluster.detectorsMask=maskCluster;
        mClusters.push_back(cluster);
        break;
    }
    case 7:
    /// CLASSES: name mask descriptor cluster LM L0 L1
    {
        CTPClass cls;
        if(ntokens != 4)
        {
            std::cout << "Syntax error in CLASSES:" << line << std::endl;
            return level;
        }
        cls.name = tokens[0];
        try
        {
            cls.classMask = std::stoull(tokens[1]);
        }
        catch(...)
        {
            std::cout << "Syntax error in CLASSES in mask:" << line << std::endl;
            return level;
        }
        std::string token=tokens[2];
        auto it = std::find_if(mDescriptors.begin(),mDescriptors.end(),[&token](const CTPDescriptor& obj){return obj.name == token;});
        if(it != mDescriptors.end())
        {
            cls.descriptor = &*it;
        }
        else
        {
            ///Internal error
        }
        ///
        token=tokens[3];
        auto it2 = std::find_if(mClusters.begin(),mClusters.end(),[&token](const CTPCluster& obj){return obj.name == token;});
        if(it2 != mClusters.end())
        {
            cls.cluster = &*it2;
        }
        else
        {
            ///INternal error
        }
        mCTPClasses.push_back(cls);
        break;
    }
    default:
    {
        std::cout << "Error: Unknown level:"<< level << std::endl;
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
  stream << "Configuration:" << mName << "\n Version:"<< mVersion <<  std::endl;
  stream << "CTP BC  masks:" << std::endl;
  for (const auto i : mBCMasks) {
    i.printStream(stream);
  }
  stream << "CTP inputs:" << std::endl;
  for (const auto i : mInputs) {
    i.printStream(stream);
  }
  stream << "CTP descriptors:" << std::endl;
  for (const auto i : mDescriptors) {
    i.printStream(stream);
  }
  stream << "CTP detectors:" << std::endl;
  for (const auto i : mDetectors) {
    i.printStream(stream);
  }
  stream << "CTP clusters:" << std::endl;
  for (const auto i : mClusters) {
    i.printStream(stream);
  }
  stream << "CTP classes:" << std::endl;
  for (const auto i : mCTPClasses) {
    i.printStream(stream);
  }
}
uint64_t CTPConfiguration::getInputMask(const std::string& name)
{
  for(auto const& inp: mInputs) {
    if(inp.name == name) {
        return inp.inputMask;
      }
  }
  return 0;
}
bool CTPConfiguration::isMaskInInputs(uint64_t& mask)
{
  for(auto const& inp: mInputs) {
     if(inp.inputMask == mask) return true;
  }
  return false;
}
