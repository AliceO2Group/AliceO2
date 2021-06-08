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

using namespace o2::ctp;

void BCMask::printStream(std::ostream& stream) const
{
  stream << "CTP BC mask:" << name << std::endl;
  /// <<  ":" << BCmask << std::endl;
}
void CTPInput::printStream(std::ostream& stream) const
{
  stream << "CTP Input:" << name << "Detector:" << detID << " Hardware mask:" << inputMask << std::endl;
}
void CTPDescriptor::printStream(std::ostream& stream) const
{
  stream << "CTP Descriptor:" << name << " Inputs:" << inputsMask << std::endl;
}
void CTPDetector::printStream(std::ostream& stream) const
{
  o2::detectors::DetID det(detID);
  stream << "CTP Detector:" << det.getName() << " HBaccepted:" << HBaccepted << std::endl;
}
void CTPCluster::printStream(std::ostream& stream) const
{
  stream << "CTP Cluster:" << name << " Inputs:" << detectorsMask << std::endl;
}
void CTPClass::printStream(std::ostream& stream) const
{
  stream << "CTP Class:" << name << " Hardware mask:" << classMask << std::endl;
}
/// CTP configuration
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
  stream << "Configuration:" << mName << std::endl;
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
