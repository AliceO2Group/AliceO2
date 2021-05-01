// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsCTP/Configuration.h"
#include <iostream>

using namespace o2::ctp;

void CTPInput::printStream(std::ostream& stream) const
{
  stream << "CTP Input:" << mName << " Hardware mask:" << mInputMask << std::endl;
}
void CTPDescriptor::printStream(std::ostream& stream) const
{
  stream << "CTP Descriptor:" << mName << std::endl;
}
void CTPClass::printStream(std::ostream& stream) const
{
  stream << "CTP Class:" << mName << " Hardware mask:" << std::endl;
}
void CTPConfiguration::addCTPClass(CTPClass& ctpclass)
{
  CTPClasses.push_back(ctpclass);
}
