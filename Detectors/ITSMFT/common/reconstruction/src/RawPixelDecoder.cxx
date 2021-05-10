// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file RawPixelDecoder.cxx
/// \brief Alpide pixel reader for raw data processing

#include "DetectorsRaw/RDHUtils.h"
#include "ITSMFTReconstruction/RawPixelDecoder.h"
#include "DPLUtils/DPLRawParser.h"
#include "Framework/InputRecordWalker.h"
#include "CommonUtils/StringUtils.h"

#ifdef WITH_OPENMP
#include <omp.h>
#endif

using namespace o2::itsmft;
using namespace o2::framework;
using RDHUtils = o2::raw::RDHUtils;

///______________________________________________________________
/// C-tor
template <class Mapping>
RawPixelDecoder<Mapping>::RawPixelDecoder()
{
  mRUEntry.fill(-1); // no known links in the beginning
  mTimerTFStart.Stop();
  mTimerDecode.Stop();
  mTimerFetchData.Stop();
  mSelfName = o2::utils::Str::concat_string(Mapping::getName(), "Decoder");
}

///______________________________________________________________
///
template <class Mapping>
void RawPixelDecoder<Mapping>::printReport(bool decstat, bool skipNoErr) const
{
  LOGF(INFO, "%s Decoded %zu hits in %zu non-empty chips in %u ROFs with %d threads", mSelfName, mNPixelsFired, mNChipsFired, mROFCounter, mNThreads);
  double cpu = 0, real = 0;
  auto& tmrS = const_cast<TStopwatch&>(mTimerTFStart);
  LOGF(INFO, "%s Timing Start TF:  CPU = %.3e Real = %.3e in %d slots", mSelfName, tmrS.CpuTime(), tmrS.RealTime(), tmrS.Counter() - 1);
  cpu += tmrS.CpuTime();
  real += tmrS.RealTime();
  auto& tmrD = const_cast<TStopwatch&>(mTimerDecode);
  LOGF(INFO, "%s Timing Decode:    CPU = %.3e Real = %.3e in %d slots", mSelfName, tmrD.CpuTime(), tmrD.RealTime(), tmrD.Counter() - 1);
  cpu += tmrD.CpuTime();
  real += tmrD.RealTime();
  auto& tmrF = const_cast<TStopwatch&>(mTimerFetchData);
  LOGF(INFO, "%s Timing FetchData: CPU = %.3e Real = %.3e in %d slots", mSelfName, tmrF.CpuTime(), tmrF.RealTime(), tmrF.Counter() - 1);
  cpu += tmrF.CpuTime();
  real += tmrF.RealTime();
  LOGF(INFO, "%s Timing Total:     CPU = %.3e Real = %.3e in %d slots in %s mode", mSelfName, cpu, real, tmrS.Counter() - 1,
       mDecodeNextAuto ? "AutoDecode" : "ExternalCall");

  if (decstat) {
    LOG(INFO) << "GBT Links decoding statistics" << (skipNoErr ? " (only links with errors are reported)" : "");
    for (auto& lnk : mGBTLinks) {
      lnk.statistics.print(skipNoErr);
      lnk.chipStat.print(skipNoErr);
    }
  }
}

///______________________________________________________________
/// Decode next trigger for all links
template <class Mapping>
int RawPixelDecoder<Mapping>::decodeNextTrigger()
{
  mTimerDecode.Start(false);
  mNChipsFiredROF = 0;
  mNPixelsFiredROF = 0;
  mInteractionRecord.clear();
  int nLinksWithData = 0, nru = mRUDecodeVec.size();
  do {
#ifdef WITH_OPENMP
    omp_set_num_threads(mNThreads);
#pragma omp parallel for schedule(dynamic) reduction(+ \
                                                     : nLinksWithData, mNChipsFiredROF, mNPixelsFiredROF)
#endif
    for (int iru = 0; iru < nru; iru++) {
      nLinksWithData += decodeNextTrigger(iru);
      mNChipsFiredROF += mRUDecodeVec[iru].nChipsFired;
      int npix = 0;
      for (int ic = mRUDecodeVec[iru].nChipsFired; ic--;) {
        npix += mRUDecodeVec[iru].chipsData[ic].getData().size();
      }
      mNPixelsFiredROF += npix;
    }

    if (nLinksWithData) { // fill some statistics
      mROFCounter++;
      mNChipsFired += mNChipsFiredROF;
      mNPixelsFired += mNPixelsFiredROF;
      mCurRUDecodeID = 0; // getNextChipData will start from here
      mLastReadChipID = -1;
      // set IR and trigger from the 1st non empty link
      for (const auto& link : mGBTLinks) {
        if (link.status == GBTLink::DataSeen) {
          mInteractionRecord = link.ir;
          mInteractionRecordHB = o2::raw::RDHUtils::getHeartBeatIR(*link.lastRDH);
          mTrigger = link.trigger;
          break;
        }
      }
      break;
    }

  } while (mNLinksDone < mGBTLinks.size());

  ensureChipOrdering();
  mTimerDecode.Stop();
  return nLinksWithData;
}

///______________________________________________________________
/// prepare for new TF
template <class Mapping>
void RawPixelDecoder<Mapping>::startNewTF(InputRecord& inputs)
{
  mTimerTFStart.Start(false);
  for (auto& link : mGBTLinks) {
    link.lastRDH = nullptr;  // pointers will be invalid
    link.clear(false, true); // clear data but not the statistics
  }
  for (auto& ru : mRUDecodeVec) {
    ru.clear();
  }
  setupLinks(inputs);
  mNLinksDone = 0;
  mTimerTFStart.Stop();
}

///______________________________________________________________
/// Decode next trigger for given RU, return number of decoded GBT words
template <class Mapping>
int RawPixelDecoder<Mapping>::decodeNextTrigger(int iru)
{
  auto& ru = mRUDecodeVec[iru];
  ru.clear();
  int ndec = 0; // number of yet non-empty links
  for (int il = 0; il < RUDecodeData::MaxLinksPerRU; il++) {
    auto* link = getGBTLink(ru.links[il]);
    if (link) {
      auto res = link->collectROFCableData(mMAP);
      if (res == GBTLink::DataSeen) { // at the moment process only DataSeen
        ndec++;
      } else if (res == GBTLink::StoppedOnEndOfData || res == GBTLink::AbortedOnError) { // this link has exhausted its data or it has to be discarded due to the error
        mNLinksDone++;
      }
    }
  }
  if (ndec) {
    ru.decodeROF(mMAP);
  }
  return ndec;
}

///______________________________________________________________
/// Setup links checking the very RDH of every input
template <class Mapping>
void RawPixelDecoder<Mapping>::setupLinks(InputRecord& inputs)
{
  mCurRUDecodeID = NORUDECODED;
  auto nLinks = mGBTLinks.size();
  auto origin = (mUserDataOrigin == o2::header::gDataOriginInvalid) ? mMAP.getOrigin() : mUserDataOrigin;
  auto datadesc = (mUserDataDescription == o2::header::gDataDescriptionInvalid) ? o2::header::gDataDescriptionRawData : mUserDataDescription;
  std::vector<InputSpec> filter{InputSpec{"filter", ConcreteDataTypeMatcher{origin, datadesc}, Lifetime::Timeframe}};

  // if we see requested data type input with 0xDEADBEEF subspec and 0 payload this means that the "delayed message"
  // mechanism created it in absence of real data from upstream. Processor should send empty output to not block the workflow
  {
    std::vector<InputSpec> dummy{InputSpec{"dummy", ConcreteDataMatcher{origin, datadesc, 0xDEADBEEF}}};
    for (const auto& ref : InputRecordWalker(inputs, dummy)) {
      const auto dh = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      if (dh->payloadSize == 0) {
        LOGP(WARNING, "Found input [{}/{}/{:#x}] TF#{} 1st_orbit:{} Payload {} : assuming no payload for all links in this TF",
             dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, dh->tfCounter, dh->firstTForbit, dh->payloadSize);
        return;
      }
    }
  }

  DPLRawParser parser(inputs, filter);
  uint32_t currSSpec = 0xffffffff; // dummy starting subspec
  int linksAdded = 0;
  for (auto it = parser.begin(); it != parser.end(); ++it) {
    auto const* dh = it.o2DataHeader();
    auto& lnkref = mSubsSpec2LinkID[dh->subSpecification];
    const auto& rdh = *reinterpret_cast<const header::RDHAny*>(it.raw()); // RSTODO this is a hack in absence of generic header getter

    if (lnkref.entry == -1) { // new link needs to be added
      lnkref.entry = int(mGBTLinks.size());
      auto& lnk = mGBTLinks.emplace_back(RDHUtils::getCRUID(rdh), RDHUtils::getFEEID(rdh), RDHUtils::getEndPointID(rdh), RDHUtils::getLinkID(rdh), lnkref.entry);
      getCreateRUDecode(mMAP.FEEId2RUSW(RDHUtils::getFEEID(rdh))); // make sure there is a RU for this link
      lnk.verbosity = GBTLink::Verbosity(mVerbosity);
      lnk.format = mFormat;
      LOG(INFO) << mSelfName << " registered new link " << lnk.describe() << " RUSW=" << int(mMAP.FEEId2RUSW(lnk.feeID));
      linksAdded++;
    }
    auto& link = mGBTLinks[lnkref.entry];
    if (currSSpec != dh->subSpecification) { // this is the 1st part for this link in this TF, next parts must follow contiguously!!!
      link.clear(false, true);               // clear link data except statistics
      currSSpec = dh->subSpecification;
    }
    link.cacheData(it.raw(), RDHUtils::getMemorySize(rdh));
  }

  if (linksAdded) { // new links were added, update link<->RU mapping, usually is done for 1st TF only
    if (nLinks) {
      LOG(WARNING) << mSelfName << " New links appeared although the initialization was already done";
      for (auto& ru : mRUDecodeVec) { // reset RU->link references since they may have been changed
        memset(&ru.links[0], -1, RUDecodeData::MaxLinksPerRU * sizeof(int));
      }
    }
    // sort RUs in stave increasing order
    std::sort(mRUDecodeVec.begin(), mRUDecodeVec.end(), [](const RUDecodeData& ruA, const RUDecodeData& ruB) -> bool { return ruA.ruSWID < ruB.ruSWID; });
    for (auto i = 0; i < mRUDecodeVec.size(); i++) {
      mRUEntry[mRUDecodeVec[i].ruSWID] = i;
    }
    nLinks = mGBTLinks.size();
    // attach link to corresponding RU: this can be done once all RUs are created, to make sure their pointers don't change
    for (int il = 0; il < nLinks; il++) {
      auto& link = mGBTLinks[il];
      bool newLinkAdded = (link.ruPtr == nullptr);
      link.ruPtr = getRUDecode(mMAP.FEEId2RUSW(link.feeID)); // link to RU reference, reattach even it was already set before
      uint16_t lr, ruOnLr, linkInRU;
      mMAP.expandFEEId(link.feeID, lr, ruOnLr, linkInRU);
      if (newLinkAdded) {
        LOG(INFO) << mSelfName << " Attaching " << link.describe() << " to RU#" << int(mMAP.FEEId2RUSW(link.feeID))
                  << " (stave " << ruOnLr << " of layer " << lr << ')';
      }
      link.idInRU = linkInRU;
      link.ruPtr->links[linkInRU] = il; // RU to link reference
    }
  }
}

///______________________________________________________________
/// get RU decode container for RU with given SW ID, if does not exist, create it
template <class Mapping>
RUDecodeData& RawPixelDecoder<Mapping>::getCreateRUDecode(int ruSW)
{
  assert(ruSW < mMAP.getNRUs());
  if (mRUEntry[ruSW] < 0) {
    mRUEntry[ruSW] = mRUDecodeVec.size();
    auto& ru = mRUDecodeVec.emplace_back();
    ru.ruSWID = ruSW;
    ru.ruInfo = mMAP.getRUInfoSW(ruSW); // info on the stave/RU
    ru.chipsData.resize(mMAP.getNChipsOnRUType(ru.ruInfo->ruType));
    LOG(INFO) << mSelfName << " Defining container for RU " << ruSW << " at slot " << mRUEntry[ruSW];
  }
  return mRUDecodeVec[mRUEntry[ruSW]];
}

///______________________________________________________________________
template <class Mapping>
ChipPixelData* RawPixelDecoder<Mapping>::getNextChipData(std::vector<ChipPixelData>& chipDataVec)
{
  // decode new RU if no cached non-empty chips
  for (; mCurRUDecodeID < mRUDecodeVec.size(); mCurRUDecodeID++) {
    auto& ru = mRUDecodeVec[mCurRUDecodeID];
    if (ru.lastChipChecked < ru.nChipsFired) {
      auto& chipData = ru.chipsData[ru.lastChipChecked++];
      assert(mLastReadChipID < chipData.getChipID());
      mLastReadChipID = chipData.getChipID();
      chipDataVec[mLastReadChipID].swap(chipData);
      return &chipDataVec[mLastReadChipID];
    }
  }
  // will need to decode new trigger
  if (!mDecodeNextAuto || !decodeNextTrigger()) { // no more data to decode
    return nullptr;
  }
  return getNextChipData(chipDataVec);
}

///______________________________________________________________________
template <class Mapping>
bool RawPixelDecoder<Mapping>::getNextChipData(ChipPixelData& chipData)
{
  /// read single chip data to the provided container
  for (; mCurRUDecodeID < mRUDecodeVec.size(); mCurRUDecodeID++) {
    auto& ru = mRUDecodeVec[mCurRUDecodeID];
    if (ru.lastChipChecked < ru.nChipsFired) {
      auto& ruchip = ru.chipsData[ru.lastChipChecked++];
      assert(mLastReadChipID < chipData.getChipID());
      mLastReadChipID = chipData.getChipID();
      chipData.swap(ruchip);
      return true;
    }
  }
  // will need to decode new trigger
  if (!mDecodeNextAuto || !decodeNextTrigger()) { // no more data to decode
    return false;
  }
  return getNextChipData(chipData); // is it ok to use recursion here?
}

///______________________________________________________________________
template <>
void RawPixelDecoder<ChipMappingMFT>::ensureChipOrdering()
{
  // define looping order, if mCurRUDecodeID < mRUDecodeVec.size(), this means that decodeNextTrigger() was called before
  if (mCurRUDecodeID < mRUDecodeVec.size()) { // define sort order
    for (; mCurRUDecodeID < mRUDecodeVec.size(); mCurRUDecodeID++) {
      auto& ru = mRUDecodeVec[mCurRUDecodeID];
      while (ru.lastChipChecked < ru.nChipsFired) {
        mOrderedChipsPtr.push_back(&ru.chipsData[ru.lastChipChecked++]);
      }
    }
    // sort in decreasing order
    std::sort(mOrderedChipsPtr.begin(), mOrderedChipsPtr.end(), [](const ChipPixelData* a, const ChipPixelData* b) { return a->getChipID() > b->getChipID(); });
  }
}

///______________________________________________________________________
template <>
ChipPixelData* RawPixelDecoder<ChipMappingMFT>::getNextChipData(std::vector<ChipPixelData>& chipDataVec)
{
  if (!mOrderedChipsPtr.empty()) {
    auto chipData = *mOrderedChipsPtr.back();
    assert(mLastReadChipID < chipData.getChipID());
    mLastReadChipID = chipData.getChipID();
    chipDataVec[mLastReadChipID].swap(chipData);
    mOrderedChipsPtr.pop_back();
    return &chipDataVec[mLastReadChipID];
  }
  // will need to decode new trigger
  if (!mDecodeNextAuto || !decodeNextTrigger()) { // no more data to decode
    return nullptr;
  }
  return getNextChipData(chipDataVec);
}

///______________________________________________________________________
template <>
bool RawPixelDecoder<ChipMappingMFT>::getNextChipData(ChipPixelData& chipData)
{
  if (!mOrderedChipsPtr.empty()) {
    auto ruChip = *mOrderedChipsPtr.back();
    assert(mLastReadChipID < ruChip.getChipID());
    mLastReadChipID = ruChip.getChipID();
    ruChip.swap(chipData);
    mOrderedChipsPtr.pop_back();
    return true;
  }
  // will need to decode new trigger
  if (!mDecodeNextAuto || !decodeNextTrigger()) { // no more data to decode
    return false;
  }
  return getNextChipData(chipData); // is it ok to use recursion here?
}

///______________________________________________________________________
template <class Mapping>
void RawPixelDecoder<Mapping>::setVerbosity(int v)
{
  mVerbosity = v;
  for (auto& link : mGBTLinks) {
    link.verbosity = GBTLink::Verbosity(v);
  }
}

///______________________________________________________________________
template <class Mapping>
void RawPixelDecoder<Mapping>::setNThreads(int n)
{
#ifdef WITH_OPENMP
  mNThreads = n > 0 ? n : 1;
#else
  LOG(WARNING) << mSelfName << " Multithreading is not supported, imposing single thread";
  mNThreads = 1;
#endif
}

///______________________________________________________________________
template <class Mapping>
void RawPixelDecoder<Mapping>::setFormat(GBTLink::Format f)
{
  assert(int(f) >= 0 && int(f) < GBTLink::NFormats);
  mFormat = f;
}

///______________________________________________________________________
template <class Mapping>
void RawPixelDecoder<Mapping>::clearStat()
{
  // clear statistics
  for (auto& lnk : mGBTLinks) {
    lnk.clear(true, false);
  }
}

template class o2::itsmft::RawPixelDecoder<o2::itsmft::ChipMappingITS>;
template class o2::itsmft::RawPixelDecoder<o2::itsmft::ChipMappingMFT>;
