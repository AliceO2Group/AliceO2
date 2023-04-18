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

/// \file IRFrameSelector.cxx
/// \brief Class to check if give InteractionRecord or IRFrame is selected by the external IRFrame vector
/// \author ruben.shahoyan@cern.ch

#include "CommonUtils/IRFrameSelector.h"
#include "Framework/Logger.h"
#include <TFile.h>
#include <TTree.h>
#include <TKey.h>

using namespace o2::utils;

gsl::span<const o2::dataformats::IRFrame> IRFrameSelector::getMatchingFrames(const o2::dataformats::IRFrame& fr)
{
  // extract span of IRFrames matching to fr
  auto lower = std::lower_bound(mFrames.begin(), mFrames.end(), o2::dataformats::IRFrame{fr.getMin(), fr.getMin()});
  if (lower == mFrames.end() || *lower > fr) {
    return {};
  }
  auto upper = std::upper_bound(lower, mFrames.end(), o2::dataformats::IRFrame{fr.getMax(), fr.getMax()});
  return {&*lower, size_t(std::distance(lower, upper))};
}

long IRFrameSelector::check(o2::dataformats::IRFrame fr, size_t bwd, size_t fwd)
{
  // check if fr overlaps with at least 1 entry in the frames container, if needed expand fr by -bwd and fwd BCs from 2 sides
  long ans = -1;
  if (bwd) {
    fr.setMin(fr.getMin().toLong() > bwd ? fr.getMin() - bwd : o2::InteractionRecord{0, 0});
  }
  if (fwd) {
    fr.setMax(o2::InteractionRecord::MaxGlobalBCs - fr.getMax().toLong() > fwd ? fr.getMax() + fwd : o2::InteractionRecord::getIRMaxBC());
  }
  // find entry which overlaps or above fr
  auto fullcheck = [&fr, this]() -> long {
    auto lower = std::lower_bound(this->mFrames.begin(), this->mFrames.end(), fr);
    if (lower == this->mFrames.end()) {
      this->mLastBoundID = this->mFrames.size();
    } else {
      this->mLastBoundID = std::distance(this->mFrames.begin(), lower);
      return fr.isOutside(*lower) == o2::dataformats::IRFrame::Inside ? this->mLastBoundID : -1;
    }
    return -1;
  };

  if (mLastBoundID < 0) { // 1st call, do full search
    ans = fullcheck();
  } else { // we assume that the new query is in the vicinity of the previous one
    int steps = 0;
    constexpr int MaxSteps = 3;
    if (fr.getMin() >= mLastIRFrameChecked.getMin()) { // check entries including and following mLastBoundID
      for (; mLastBoundID < (long)mFrames.size(); mLastBoundID++) {
        auto res = fr.isOutside(mFrames[mLastBoundID]);
        if (res == o2::dataformats::IRFrame::Above) {
          break; // no point in checking further
        } else if (res == o2::dataformats::IRFrame::Inside) {
          ans = mLastBoundID;
          break;
        }
        if (++steps == MaxSteps) {
          ans = fullcheck();
          break;
        }
      }
    } else { // check entries preceding mLastBoundID
      if (mLastBoundID >= (long)mFrames.size()) {
        mLastBoundID = (long)mFrames.size() - 1;
      }
      for (; mLastBoundID >= 0; mLastBoundID--) {
        auto res = fr.isOutside(mFrames[mLastBoundID]);
        if (res == o2::dataformats::IRFrame::Below) {
          mLastBoundID++; // no point in checking further
          break;
        } else if (res == o2::dataformats::IRFrame::Inside) {
          ans = mLastBoundID;
          break;
        }
        if (++steps == MaxSteps) {
          ans = fullcheck();
          break;
        }
      }
      if (mLastBoundID < 0) {
        mLastBoundID = 0;
      }
    }
  }
  mLastIRFrameChecked = fr;
  return ans;
}

size_t IRFrameSelector::loadIRFrames(const std::string& fname)
{
  // read IRFrames to filter from the file
  std::unique_ptr<TFile> tfl(TFile::Open(fname.c_str()));
  if (!tfl || tfl->IsZombie()) {
    LOGP(fatal, "Cannot open selected IRFrames file {}", fname);
  }
  auto klst = gDirectory->GetListOfKeys();
  TIter nextkey(klst);
  TKey* key;
  std::string clVec{TClass::GetClass("std::vector<o2::dataformats::IRFrame>")->GetName()};
  bool done = false;
  bool toBeSorted = false;
  while ((key = (TKey*)nextkey())) {
    std::string kcl(key->GetClassName());
    if (kcl == clVec) {
      auto* v = (std::vector<o2::dataformats::IRFrame>*)tfl->GetObjectUnchecked(key->GetName());
      if (!v) {
        LOGP(fatal, "Failed to extract IRFrames vector {} from {}", key->GetName(), fname);
      }
      mOwnList.insert(mOwnList.end(), v->begin(), v->end());
      LOGP(info, "Loaded {} IRFrames from vector {} of {}", mOwnList.size(), key->GetName(), fname);
      done = true;
      break;
    } else if (kcl == "TTree") {
      std::unique_ptr<TTree> tr((TTree*)tfl->Get(key->GetName()));
      if (!tr->GetBranch("IRFrames")) {
        LOGP(fatal, "Did not find branch IRFrames in the tree {}", key->GetName());
      }
      std::vector<o2::dataformats::IRFrame> tmpv, *tpmvPtr = &tmpv;
      tr->SetBranchAddress("IRFrames", &tpmvPtr);
      for (int i = 0; i < (int)tr->GetEntries(); i++) {
        tr->GetEntry(i);
        mOwnList.insert(mOwnList.end(), tmpv.begin(), tmpv.end());
      }
      done = true;
      LOGP(info, "Loaded {} IRFrames from tree {} of {}", mOwnList.size(), key->GetName(), fname);
      break;
    } else if (kcl == "TDirectoryFile") {
      TTree* bcRanges = (TTree*)tfl->Get(fmt::format("{}/O2bcranges", key->GetName()).data());
      if (!bcRanges) {
        continue;
      }
      LOGP(info, "Loading BCrange trees in the directory {}", key->GetName());
      ULong64_t minBC, maxBC;
      bcRanges->SetBranchAddress("fBCstart", &minBC);
      bcRanges->SetBranchAddress("fBCend", &maxBC);
      for (int i = 0; i < (int)bcRanges->GetEntries(); i++) {
        bcRanges->GetEntry(i);
        mOwnList.emplace_back(InteractionRecord::long2IR(minBC), InteractionRecord::long2IR(maxBC));
      }
      done = true;
      toBeSorted = true;
    }
  }

  if (toBeSorted) {
    LOGP(info, "Sorting {} IRFrames", mOwnList.size());
    std::sort(mOwnList.begin(), mOwnList.end(), [](const auto& a, const auto& b) { return a.getMin() < b.getMin(); });
  }
  if (!true) {
    LOGP(fatal, "Did not find neither tree nor vector of IRFrames in {}", fname);
  }
  setSelectedIRFrames(mOwnList);
  return mOwnList.size();
}

void IRFrameSelector::print(bool lst) const
{
  LOGP(info, "Last query stopped at entry {} for IRFrame {}:{}", mLastBoundID,
       mLastIRFrameChecked.getMin().asString(), mLastIRFrameChecked.getMax().asString());
  if (lst) {
    size_t i = 0;
    for (const auto& f : mFrames) {
      LOGP(info, "Frame#{}: {}:{}", i++, f.getMin().asString(), f.getMax().asString());
    }
  }
}

void IRFrameSelector::clear()
{
  mIsSet = false;
  mOwnList.clear();
  mFrames = {};
}

void IRFrameSelector::applyMargins(size_t bwd, size_t fwd, long shift, bool removeOverlaps)
{
  // apply shift and margin to all IRFrames by converting them to IRFrame.getMin() - fwd + shift, IRFrame.getMax() + bwd + shift
  LOGP(debug, "applyMargins({},{},{})", bwd, fwd, removeOverlaps);
  if ((!fwd && !bwd && !shift) || !mIsSet || !mFrames.size()) {
    return;
  }
  mLastBoundID = -1;
  std::vector<o2::dataformats::IRFrame> lst;
  long shiftBwd = bwd - shift, shiftFwd = fwd + shift;

  for (const auto& fr : mFrames) {
    auto irmin = fr.getMin();
    auto irmax = fr.getMax();
    if (shiftBwd > 0) {
      irmin = fr.getMin().toLong() > shiftBwd ? fr.getMin() - shiftBwd : o2::InteractionRecord{0, 0};
    } else {
      irmin = (o2::InteractionRecord::MaxGlobalBCs - fr.getMin().toLong()) > -shiftBwd ? fr.getMin() - shiftBwd : o2::InteractionRecord::getIRMaxBC();
    }
    if (shiftFwd > 0) {
      irmax = (o2::InteractionRecord::MaxGlobalBCs - fr.getMax().toLong()) > shiftFwd ? fr.getMax() + shiftFwd : o2::InteractionRecord::getIRMaxBC();
    } else {
      irmax = fr.getMax().toLong() > -shiftBwd ? fr.getMax() + shiftBwd : o2::InteractionRecord{0, 0};
    }
    LOGP(debug, "before removerlap: {}:{} -> {}:{}", fr.getMin().toLong(), fr.getMax().toLong(), irmin.toLong(), irmax.toLong());
    if (removeOverlaps && lst.size() && lst.back().getMax() >= irmin) {
      lst.back().setMax(irmax);
    } else {
      lst.emplace_back(irmin, irmax).info = fr.info;
    }
  }
  mOwnList.swap(lst);
  mFrames = gsl::span<const o2::dataformats::IRFrame>(mOwnList.data(), mOwnList.size());
}
