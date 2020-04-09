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

using namespace o2::itsmft;
using namespace o2::framework;

///______________________________________________________________
/// C-tor
template <class Mapping>
RawPixelDecoder<Mapping>::RawPixelDecoder()
{
  mRUEntry.fill(-1); // no known links in the beginning
  mTimerTFStart.Stop();
  mTimerDecode.Stop();
  mTimeFetchData.Stop();
}

///______________________________________________________________
/// D-tor
template <class Mapping>
RawPixelDecoder<Mapping>::~RawPixelDecoder()
{
  printReport();
}

///______________________________________________________________
/// print timing, cannot be const since TStopwatch getters are not const...
template <class Mapping>
void RawPixelDecoder<Mapping>::printReport()
{
  LOGF(INFO, "Decoded %zu hits in %zu non-empty chips in %u ROFs", mNPixelsFired, mNChipsFired, mROFCounter);
  double cpu = 0, real = 0;
  LOGF(INFO, "Timing Start TF:  CPU = %.3e Real = %.3e in %d slots", mTimerTFStart.CpuTime(), mTimerTFStart.RealTime(), mTimerTFStart.Counter());
  cpu += mTimerTFStart.CpuTime();
  real += mTimerTFStart.RealTime();
  LOGF(INFO, "Timing Decode:    CPU = %.3e Real = %.3e in %d slots", mTimerDecode.CpuTime(), mTimerDecode.RealTime(), mTimerDecode.Counter());
  cpu += mTimerDecode.CpuTime();
  real += mTimerDecode.RealTime();
  LOGF(INFO, "Timing FetchData: CPU = %.3e Real = %.3e in %d slots", mTimeFetchData.CpuTime(), mTimeFetchData.RealTime(), mTimeFetchData.Counter());
  cpu += mTimeFetchData.CpuTime();
  real += mTimeFetchData.RealTime();
  LOGF(INFO, "Timing Total:     CPU = %.3e Real = %.3e", cpu, real);
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
  for (int iru = 0; iru < nru; iru++) {
    nLinksWithData += decodeNextTrigger(iru);
    mNChipsFiredROF += mRUDecodeVec[iru].nChipsFired;
    for (int ic = mRUDecodeVec[iru].nChipsFired; ic--;) {
      mNPixelsFiredROF += mRUDecodeVec[iru].chipsData[ic].getData().size();
    }
  }
  if (nLinksWithData) { // fill some statistics
    mROFCounter++;
    mNChipsFired += mNChipsFiredROF;
    mNPixelsFired += mNPixelsFiredROF;
    mCurRUDecodeID = 0; // getNextChipData will start from here
  }
  mTimerDecode.Stop();
  // LOG(INFO) << "Chips Fired: " << mNChipsFiredROF << " NPixels: " << mNPixelsFiredROF << " at IR " << mInteractionRecord << " of HBF " << mInteractionRecordHB;
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
        if (mInteractionRecord > link->ir) { // update interaction record RSTOD: do we need to set it for every chip?
          mInteractionRecord = link->ir;
          mInteractionRecordHB = o2::raw::RDHUtils::getHBIR(*link->lastRDH);
          mTrigger = link->trigger;
        }
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
  std::vector<InputSpec> filter{InputSpec{"filter", ConcreteDataTypeMatcher{mMAP.getOrigin(), "RAWDATA"}, Lifetime::Timeframe}};
  DPLRawParser parser(inputs, filter);
  uint32_t currSSpec = 0xffffffff; // dummy starting subspec
  int linksAdded = 0;
  for (auto it = parser.begin(); it != parser.end(); ++it) {
    auto const* dh = it.o2DataHeader();
    auto& lnkref = mSubsSpec2LinkID[dh->subSpecification];
    const auto* rdh = it.get_if<RDH>();

    if (lnkref.entry == -1) { // new link needs to be added
      lnkref.entry = int(mGBTLinks.size());
      auto& lnk = mGBTLinks.emplace_back(rdh->cruID, rdh->feeId, rdh->endPointID, rdh->linkID, lnkref.entry);
      getCreateRUDecode(mMAP.FEEId2RUSW(rdh->feeId)); // make sure there is a RU for this link
      lnk.verbosity = mVerbosity;
      LOG(INFO) << "registered new link " << lnk.describe() << " RUSW=" << int(mMAP.FEEId2RUSW(lnk.feeID));
      linksAdded++;
    }
    auto& link = mGBTLinks[lnkref.entry];
    if (currSSpec != dh->subSpecification) { // this is the 1st part for this link in this TF, next parts must follow contiguously!!!
      link.clear(false, true);               // clear link data except statistics
      currSSpec = dh->subSpecification;
    }
    link.cacheData(it.raw(), rdh->memorySize);
  }

  if (linksAdded) { // new links were added, update link<->RU mapping, usually is done for 1st TF only
    if (nLinks) {
      LOG(WARNING) << "New links appeared although the initialization was already done";
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
        LOG(INFO) << "Attaching " << link.describe() << " to RU#" << int(mMAP.FEEId2RUSW(link.feeID))
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
    LOG(INFO) << "Defining container for RU " << ruSW << " at slot " << mRUEntry[ruSW];
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
      int id = chipData.getChipID();
      chipDataVec[id].swap(chipData);
      return &chipDataVec[id];
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
      chipData.swap(ru.chipsData[ru.lastChipChecked++]);
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
template <class Mapping>
void RawPixelDecoder<Mapping>::setVerbosity(int v)
{
  mVerbosity = v;
  for (auto& link : mGBTLinks) {
    link.verbosity = v;
  }
}

template class o2::itsmft::RawPixelDecoder<o2::itsmft::ChipMappingITS>;
template class o2::itsmft::RawPixelDecoder<o2::itsmft::ChipMappingMFT>;
