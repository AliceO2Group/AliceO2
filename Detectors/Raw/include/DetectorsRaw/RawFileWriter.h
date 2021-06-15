// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef DETECTOR_BASE_RAWFILEWRITER_H
#define DETECTOR_BASE_RAWFILEWRITER_H

/// @file   RawFileWriter.cxx
/// @author ruben.shahoyan@cern.ch
/// @brief  Utility class to write detectors data to (multiple) raw data file(s) respecting CRU format

#include <gsl/span>
#include <unordered_map>
#include <vector>
#include <map>
#include <string>
#include <string_view>
#include <functional>
#include <mutex>

#include <Rtypes.h>
#include <TTree.h>
#include <TStopwatch.h>
#include "Headers/RAWDataHeader.h"
#include "Headers/DataHeader.h"
#include "Headers/DAQID.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsRaw/RDHUtils.h"

namespace o2
{
namespace raw
{

class RawFileWriter
{

 public:
  using RDHAny = o2::header::RDHAny;
  using IR = o2::InteractionRecord;
  using CarryOverCallBack = std::function<int(const RDHAny* rdh, const gsl::span<char> data,
                                              const char* ptr, int size, int splitID,
                                              std::vector<char>& trailer, std::vector<char>& header)>;
  using EmptyPageCallBack = std::function<void(const RDHAny* rdh, std::vector<char>& emptyHBF)>;
  using NewRDHCallBack = std::function<void(const RDHAny* rdh, bool prevEmpty, std::vector<char>& filler)>;

  ///=====================================================================================
  /// output file handler with its own lock
  struct OutputFile {
    FILE* handler = nullptr;
    std::mutex fileMtx;
    OutputFile() = default;
    OutputFile(const OutputFile& src) : handler(src.handler) {}
    OutputFile& operator=(const OutputFile& src)
    {
      if (this != &src) {
        handler = src.handler;
      }
      return *this;
    }
    void write(const char* data, size_t size);
  };
  ///=====================================================================================
  struct PayloadCache {
    bool preformatted = false;
    uint32_t trigger = 0;
    uint32_t detField = 0;
    std::vector<char> payload;
    ClassDefNV(PayloadCache, 1);
  };

  ///=====================================================================================
  /// Single GBT link helper
  struct LinkData {
    static constexpr int MarginToFlush = 10 * sizeof(RDHAny); // flush superpage if free space left <= this margin
    RDHAny rdhCopy;                                          // RDH with the running info of the last RDH seen
    IR updateIR;                                          // IR at which new HBF needs to be created
    int lastRDHoffset = -1;                               // position of last RDH in the link buffer
    bool startOfRun = true;                               // to signal if this is the 1st HBF of the run or not
    uint8_t packetCounter = 0;                            // running counter
    uint16_t pageCnt = 0;                                 // running counter
    LinkSubSpec_t subspec = 0;                            // subspec according to DataDistribution
    bool discardData = false;                             // discard data if true (e.g. desired max IR reached)
    //
    size_t nTFWritten = 0;    // number of TFs written
    size_t nRDHWritten = 0;   // number of RDHs written
    size_t nBytesWritten = 0; // number of bytes written
    //
    std::string fileName{};                // file name associated with this link
    std::vector<char> buffer;              // buffer to accumulate superpage data
    RawFileWriter* writer = nullptr;       // pointer on the parent writer

    PayloadCache cacheBuffer;         // used for caching in case of async. data input
    std::unique_ptr<TTree> cacheTree; // tree to store the cache

    std::mutex mtx;

    LinkData() = default;
    ~LinkData() = default;
    LinkData(const LinkData& src);            // due to the mutex...
    LinkData& operator=(const LinkData& src); // due to the mutex...
    void close(const IR& ir);
    void print() const;
    void addData(const IR& ir, const gsl::span<char> data, bool preformatted = false, uint32_t trigger = 0, uint32_t detField = 0);
    RDHAny* getLastRDH() { return lastRDHoffset < 0 ? nullptr : reinterpret_cast<RDHAny*>(&buffer[lastRDHoffset]); }
    int getCurrentPageSize() const { return lastRDHoffset < 0 ? -1 : int(buffer.size()) - lastRDHoffset; }
    // check if we are at the beginning of new page
    bool isNewPage() const { return getCurrentPageSize() == sizeof(RDHAny); }
    std::string describe() const;

   protected:
    void addDataInternal(const IR& ir, const gsl::span<char> data, bool preformatted = false, uint32_t trigger = 0, uint32_t detField = 0, bool checkEmpty = true);
    void openHBFPage(const RDHAny& rdh, uint32_t trigger = 0);
    void addHBFPage(bool stop = false);
    void closeHBFPage();
    void flushSuperPage(bool keepLastPage = false);
    void fillEmptyHBHs(const IR& ir, bool dataAdded);
    void addPreformattedCRUPage(const gsl::span<char> data);
    void cacheData(const IR& ir, const gsl::span<char> data, bool preformatted, uint32_t trigger = 0, uint32_t detField = 0);

    /// expand buffer by positive increment and return old size
    size_t expandBufferBy(size_t by)
    {
      auto offs = buffer.size();
      buffer.resize(offs + by);
      return offs;
    }

    /// append to the end of the buffer and return the point where appended to
    size_t pushBack(const char* ptr, size_t sz, bool keepLastOnFlash = true);

    /// add RDH to buffer. In case this requires flushing of the superpage, do not keep the previous page
    size_t pushBack(const RDHAny& rdh)
    {
      nRDHWritten++;
      return pushBack(reinterpret_cast<const char*>(&rdh), sizeof(RDHAny), false);
    }

  };
  //=====================================================================================
  // If addData was called with given IR for at least 1 link, then it should be called for all links, even it with empty payload
  // This structure will check if detector has dared to do this
  struct DetLazinessCheck {
    IR ir{};
    bool preformatted = false;
    uint32_t trigger = 0;
    uint32_t detField = 0;
    size_t irSeen = 0;
    size_t completeCount = 0;
    std::unordered_map<LinkSubSpec_t, bool> linksDone; // links for which addData was called
    void acknowledge(LinkSubSpec_t s, const IR& _ir, bool _preformatted, uint32_t _trigger, uint32_t _detField);
    void completeLinks(RawFileWriter* wr, const IR& _ir);
    void clear()
    {
      linksDone.clear();
      ir.clear();
    }
  };

  //=====================================================================================

  RawFileWriter(o2::header::DataOrigin origin = o2::header::gDataOriginInvalid, bool cru = true) : mOrigin(origin)
  {
    if (!cru) {
      setRORCDetector();
    }
  }
  ~RawFileWriter();
  void useCaching();
  void doLazinessCheck(bool v) { mDoLazinessCheck = v; }
  void writeConfFile(std::string_view origin = "FLP", std::string_view description = "RAWDATA", std::string_view cfgname = "raw.cfg", bool fullPath = true) const;
  void close();

  LinkData& registerLink(uint16_t fee, uint16_t cru, uint8_t link, uint8_t endpoint, std::string_view outFileName);

  template <typename H>
  LinkData& registerLink(const H& rdh, std::string_view outFileName)
  {
    RDHAny::sanityCheckLoose<H>();
    auto& linkData = registerLink(RDHUtils::getFEEID(rdh), RDHUtils::getCRUID(rdh), RDHUtils::getLinkID(rdh), RDHUtils::getEndPointID(rdh), outFileName);
    RDHUtils::setDetectorField(linkData.rdhCopy, RDHUtils::getDetectorField(rdh));
    return linkData;
  }

  void setOrigin(o2::header::DataOrigin origin)
  {
    mOrigin = origin;
  }

  o2::header::DataOrigin getOrigin() const { return mOrigin; }

  LinkData& getLinkWithSubSpec(LinkSubSpec_t ss);

  template <typename H>
  LinkData& getLinkWithSubSpec(const H& rdh)
  {
    RDHAny::sanityCheckLoose<H>();
    return mSSpec2Link[RDHUtils::getSubSpec(RDHUtils::getCRUID(rdh), RDHUtils::getLinkID(rdh), RDHUtils::getEndPointID(rdh), RDHUtils::getFEEID(rdh))];
  }

  void addData(uint16_t feeid, uint16_t cru, uint8_t lnk, uint8_t endpoint, const IR& ir,
               const gsl::span<char> data, bool preformatted = false, uint32_t trigger = 0, uint32_t detField = 0);

  template <typename H>
  void addData(const H& rdh, const IR& ir, const gsl::span<char> data, bool preformatted = false, uint32_t trigger = 0)
  {
    RDHAny::sanityCheckLoose<H>();
    addData(RDHUtils::getFEEID(rdh), RDHUtils::getCRUID(rdh), RDHUtils::getLinkID(rdh), RDHUtils::getEndPointID(rdh), ir, data, trigger);
  }

  void setContinuousReadout() { mROMode = Continuous; }
  void setTriggeredReadout()
  {
    mROMode = Triggered;
    setDontFillEmptyHBF(true);
  }
  void setContinuousReadout(bool v)
  {
    if (v) {
      setContinuousReadout();
    } else {
      setTriggeredReadout();
    }
  }

  bool isContinuousReadout() const { return mROMode == Continuous; }
  bool isTriggeredReadout() const { return mROMode == Triggered; }
  bool isReadOutModeSet() const { return mROMode != NotSet; }
  bool isLinkRegistered(LinkSubSpec_t ss) const { return mSSpec2Link.find(ss) != mSSpec2Link.end(); }

  void setVerbosity(int v) { mVerbosity = v; }
  int getVerbosity() const { return mVerbosity; }

  int getNOutputFiles() const { return mFName2File.size(); }
  std::string getOutputFileName(int i) const
  {
    if (i >= getNOutputFiles()) {
      return "";
    }
    auto it = mFName2File.begin();
    while (i--) {
      it++;
    }
    return it->first;
  }

  OutputFile& getOutputFileForLink(const LinkData& lnk) { return mFName2File[lnk.fileName]; }

  int getSuperPageSize() const { return mSuperPageSize; }
  void setSuperPageSize(int nbytes);

  /// get highest IR seen so far
  IR getIRMax() const;

  const HBFUtils& getHBFUtils() const { return mHBFUtils; }

  template <class T>
  void setCarryOverCallBack(const T* t)
  {
    carryOverFunc = [=](const RDHAny* rdh, const gsl::span<char> data, const char* ptr, int size, int splitID,
                        std::vector<char>& trailer, std::vector<char>& header) -> int {
      return t->carryOverMethod(rdh, data, ptr, size, splitID, trailer, header);
    };
  }

  template <class T>
  void setEmptyPageCallBack(const T* t)
  {
    emptyHBFFunc = [=](const RDHAny* rdh, std::vector<char>& toAdd) {
      t->emptyHBFMethod(rdh, toAdd);
    };
  }

  template <class T>
  void setNewRDHCallBack(const T* t)
  {
    newRDHFunc = [=](const RDHAny* rdh, bool prevEmpty, std::vector<char>& toAdd) {
      t->newRDHMethod(rdh, prevEmpty, toAdd);
    };
  }

  // This is a placeholder for the function responsible to split large payload to pieces
  // fitting 8kB CRU pages.
  // The RawFileWriter receives from the encoder the payload to format according to the CRU format
  // In case this payload size does not fit into the CRU page (this may happen even if it is
  // less than 8KB, since it might be added to already partially populated CRU page of the HBF)
  // it will write on the page only part of the payload and carry over the rest on extra page(s).
  // By default the RawFileWriter will simply chunk payload as is considers necessary, but some
  // detectors want their CRU pages to be self-consistent and in case of payload splitting they
  // add in the end of page to be closed and beginning of the new page to be opened
  // (right after the RDH) detector-specific trailer and header respectively.
  //
  // The role of this method is to suggest to writer how to split the payload:
  // If this method was set to the RawFileWriter using
  // RawFileWriter::setCarryOverCallBack(pointer_on_the_converter_class);
  // then the RawFileWriter will call it before splitting.
  //
  // It provides to the carryOverMethod method the following info:
  // rdh     : RDH of the CRU page being written
  // data    : original payload received by the RawFileWriter
  // ptr     : pointer on the data in the payload which was not yet added to the link CRU pages
  // maxSize : maximum size (multiple of 16 bytes) of the bloc starting at ptr which it can
  //           accomodate at the current CRU page (i.e. what it would write by default)
  // splitID : number of times this payload was already split, 0 at 1st call
  // trailer : what it wants to add in the end of the CRU page where the data starting from ptr
  //           will be added. The trailer is supplied as an empy vector, which carryOverMethod
  //           may populate, but its size must be multiple of 16 bytes.
  // header  : what it wants to add right after the RDH of the new CRU page before the rest of
  //           the payload (starting at ptr+actualSize) will be written
  //
  // The method mast return actual size of the bloc which can be written (<=maxSize).
  // If this method populates the trailer, it must ensure that it returns the actual size such that
  // actualSize + trailer.size() <= maxSize
  // In case returned actualSize == 0, current CRU page will be closed w/o adding anything, and new
  // query of this method will be done on the new CRU page

  int carryOverMethod(const RDHAny*, const gsl::span<char> data, const char* ptr, int maxSize, int splitID,
                      std::vector<char>& trailer, std::vector<char>& header) const
  {
    return maxSize; // do nothing
  }

  // This is a placeholder for the optional callback function to provide a detector-specific filler between
  // the 1st RDH and closing RDH of the empty HBF
  //
  // It provides to the emptyHBFMethod method the following info:
  // rdh     : RDH of the CRU page opening empty RDH
  // toAdd   : a vector (supplied empty) to be filled to a size multipe of 16 bytes
  //
  void emptyHBFMethod(const RDHAny* rdh, std::vector<char>& toAdd) const
  {
  }

  // This is a placeholder for the optional callback function to provide a detector-specific filler to be added right
  // after the page starting by RDH (might be with RDH.stop=1 !) for the normal data filling (!! not automatic open/close RDHs for empty pages)
  //
  // It provides to the newRDHMethod method the following info:
  // rdh       : RDH of the CRU page to be opened
  // prevEmpty : true is previous RDH page did not receive any data
  // toAdd     : a vector (supplied empty) to be filled to a size multipe of 16 bytes
  //
  void newRDHMethod(const RDHAny* rdh, bool prevEmpty, std::vector<char>& toAdd) const
  {
  }

  int getUsedRDHVersion() const { return mUseRDHVersion; }
  void useRDHVersion(int v)
  {
    assert(v >= RDHUtils::getVersion<o2::header::RDHLowest>() && v <= RDHUtils::getVersion<o2::header::RDHHighest>());
    mUseRDHVersion = v;
  }

  bool getDontFillEmptyHBF() const { return mDontFillEmptyHBF; }
  void setDontFillEmptyHBF(bool v) { mDontFillEmptyHBF = v; }

  bool getAddSeparateHBFStopPage() const { return mAddSeparateHBFStopPage; }
  void setAddSeparateHBFStopPage(bool v) { mAddSeparateHBFStopPage = v; }

  void setRORCDetector()
  {
    mCRUDetector = false;
    setTriggeredReadout();
    setUseRDHStop(false);
  }

  void setUseRDHStop(bool v = true)
  {
    mUseRDHStop = v;
    if (!v) {
      setAddSeparateHBFStopPage(false);
    }
  }

  void setApplyCarryOverToLastPage(bool v) { mApplyCarryOverToLastPage = v; }

  bool isRORCDetector() const { return !mCRUDetector; }
  bool isCRUDetector() const { return mCRUDetector; }
  bool isRDHStopUsed() const { return mUseRDHStop; }
  bool isCarryOverToLastPageApplied() const { return mApplyCarryOverToLastPage; }

 private:
  void fillFromCache();

  enum RoMode_t { NotSet,
                  Continuous,
                  Triggered };

  const HBFUtils& mHBFUtils = HBFUtils::Instance();
  std::unordered_map<LinkSubSpec_t, LinkData> mSSpec2Link; // mapping from subSpec to link
  std::unordered_map<std::string, OutputFile> mFName2File; // mapping from filenames to actual files

  CarryOverCallBack carryOverFunc = nullptr; // default call back for large payload splitting (does nothing)
  EmptyPageCallBack emptyHBFFunc = nullptr;  // default call back for empty HBF (does nothing)
  NewRDHCallBack newRDHFunc = nullptr;       // default call back for new page opening (does nothing)
  // options
  int mVerbosity = 0;
  o2::header::DataOrigin mOrigin = o2::header::gDataOriginInvalid;
  int mUseRDHVersion = RDHUtils::getVersion<o2::header::RAWDataHeader>(); // by default, use default version
  int mSuperPageSize = 1024 * 1024; // super page size
  bool mStartTFOnNewSPage = true;   // every TF must start on a new SPage
  bool mDontFillEmptyHBF = false;   // skipp adding empty HBFs (uness it must have TF flag)
  bool mAddSeparateHBFStopPage = true; // HBF stop is added on a separate CRU page
  bool mUseRDHStop = true;             // detector uses STOP in RDH
  bool mCRUDetector = true;            // Detector readout via CRU ( RORC if false)
  bool mApplyCarryOverToLastPage = false; // call CarryOver method also for last chunk and overwrite modified trailer

  //>> caching --------------
  bool mCachingStage = false; // signal that current data should be cached
  std::mutex mCacheFileMtx;
  std::unique_ptr<TFile> mCacheFile; // file for caching
  using CacheEntry = std::vector<std::pair<LinkSubSpec_t, size_t>>;
  std::map<IR, CacheEntry> mCacheMap;
  //<< caching -------------

  TStopwatch mTimer;
  RoMode_t mROMode = NotSet;
  IR mFirstIRAdded; // 1st IR seen
  DetLazinessCheck mDetLazyCheck{};
  bool mDoLazinessCheck = true;

  ClassDefNV(RawFileWriter, 1);
}; // namespace raw

} // namespace raw
} // namespace o2

#endif //DETECTOR_BASE_RAWFILEWRITER_H
