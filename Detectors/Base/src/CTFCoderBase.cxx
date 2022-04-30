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

void CTFCoderBase::updateTimeDependentParams(ProcessingContext& pc)
{
  if (mLoadDictFromCCDB) {
    pc.inputs().get<std::vector<char>*>("ctfdict"); // just to trigger the finaliseCCDB
  }
}
