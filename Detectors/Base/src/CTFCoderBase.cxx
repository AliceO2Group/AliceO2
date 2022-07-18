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

/// \file CTFCoderBase.cxx
/// \brief Defintions for CTFCoderBase class (support of external dictionaries)
/// \author ruben.shahoyan@cern.ch

#include "DetectorsBase/CTFCoderBase.h"
#include "Framework/ControlService.h"
#include "Framework/ProcessingContext.h"
#include "Framework/InputRecord.h"

using namespace o2::ctf;
using namespace o2::framework;

void CTFCoderBase::checkDictVersion(const CTFDictHeader& h) const
{
  if (h.isValidDictTimeStamp()) { // external dictionary was used
    if (h.isValidDictTimeStamp() && h != mExtHeader) {
      throw std::runtime_error(fmt::format("Mismatch in {} CTF dictionary: need {}, provided {}", mDet.getName(), h.asString(), mExtHeader.asString()));
    }
  }
}

// Assign version of the dictionary which will be stored in the data (including dictionary data during dictionary creation)
// In case detector CTFCoder uses non-defaul dict. version, it should redefine this method in order to assign the version
// it needs ONLY when the external dictionary is not provided
void CTFCoderBase::assignDictVersion(CTFDictHeader& h) const
{
  if (mExtHeader.isValidDictTimeStamp()) {
    h = mExtHeader;
  }
  // detector code may exten it by
  //  else {
  //    h.majorVersion = <A>;
  //    h.minorVersion = <B>;
  //  }
}

void CTFCoderBase::updateTimeDependentParams(ProcessingContext& pc)
{
  if (mLoadDictFromCCDB) {
    pc.inputs().get<std::vector<char>*>("ctfdict"); // just to trigger the finaliseCCDB
  }
}
