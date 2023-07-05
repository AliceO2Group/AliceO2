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
#include "Framework/TimingInfo.h"

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

void CTFCoderBase::updateTimeDependentParams(ProcessingContext& pc, bool askTree)
{
  setFirstTFOrbit(pc.services().get<o2::framework::TimingInfo>().firstTForbit);
  if (pc.services().get<o2::framework::TimingInfo>().globalRunNumberChanged) { // this params need to be queried only once
    if (mOpType == OpType::Decoder) {
      pc.inputs().get<o2::ctp::TriggerOffsetsParam*>(mTrigOffsBinding); // this is a configurable param
    }
    if (mLoadDictFromCCDB) {
      if (askTree) {
        pc.inputs().get<TTree*>(mDictBinding); // just to trigger the finaliseCCDB
      } else {
        pc.inputs().get<std::vector<char>*>(mDictBinding); // just to trigger the finaliseCCDB
      }
    }
  }
}

bool CTFCoderBase::isTreeDictionary(const void* buff) const
{
  // heuristic check for the dictionary being a tree
  const char* patt[] = {"ccdb_object", "ctf_dictionary"};
  const char* ptr = reinterpret_cast<const char*>(buff);
  bool found = false;
  int i = 0, np = sizeof(patt) / sizeof(char*);
  while (i < 50 && !found) {
    for (int ip = 0; ip < np; ip++) {
      const auto *p = patt[ip], *s = &ptr[i];
      while (*p && *s == *p) {
        p++;
        s++;
      }
      if (!*p) {
        found = true;
        break;
      }
    }
    i++;
  }
  return found;
}

void CTFCoderBase::reportIRFrames()
{
  static bool repDone = false;
  if (!repDone) {
    LOGP(info, "IRFrames will be selected with shift {}, forward {} margin and backward {} margin (in BCs)", mIRFrameSelShift, mIRFrameSelMarginBwd, mIRFrameSelMarginFwd);
    repDone = true;
  }
}
