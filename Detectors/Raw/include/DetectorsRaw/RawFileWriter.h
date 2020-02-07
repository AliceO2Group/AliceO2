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
#include <string>
#include <functional>

#include <Rtypes.h>
#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/HBFUtils.h"

/*
#include <cstdio>
#include <map>

#include <utility>
#include "Headers/DataHeader.h"
*/

namespace o2
{
namespace raw
{

class RawFileWriter
{

 public:
  using RDH = o2::header::RAWDataHeaderV4;
  using IR = o2::InteractionRecord;
  using CarryOverCallBack = std::function<int(const RDH& rdh, const gsl::span<char> data,
                                              const char* ptr, int size, int splitID,
                                              std::vector<char>& trailer, std::vector<char>& header)>;
  using EmptyPageCallBack = std::function<void(const RDH& rdh, std::vector<char>& emptyHBF)>;

  //=====================================================================================
  struct LinkData {
    static constexpr int MarginToFlush = 2 * sizeof(RDH); // flush superpage if free space left <= this margin
    RDH rdhCopy;                                          // RDH with the running info of the last RDH seen
    IR updateIR;                                          // IR at which new HBF needs to be created
    FILE* file = nullptr;                                 // file handler associated with this link
    int lastRDHoffset = -1;                               // position of last RDH in the link buffer
    bool startOfRun = true;                               // to signal if this is the 1st HBF of the run or not
    uint8_t packetCounter = 0;                            // running counter
    uint16_t pageCnt = 0;                                 // running counter
    LinkSubSpec_t subspec = 0;                            // subspec according to DataDistribution
    //
    size_t nTFWritten = 0;    // number of TFs written
    size_t nRDHWritten = 0;   // number of RDHs written
    size_t nBytesWritten = 0; // number of bytes written
    //
    std::string fileName{};                // file name associated with this link
    std::vector<char> buffer;              // buffer to accumulate superpage data
    const RawFileWriter* writer = nullptr; // pointer on the parent writer

    LinkData() = default;
    ~LinkData();
    void close(const IR& ir);
    void print() const;
    void addData(const IR& ir, const gsl::span<char> data);
    RDH* getLastRDH() { return lastRDHoffset < 0 ? nullptr : reinterpret_cast<RDH*>(&buffer[lastRDHoffset]); }
    std::string describe() const;

   protected:
    void openHBFPage(const RDH& rdh);
    void addHBFPage(bool stop = false);
    void closeHBFPage() { addHBFPage(true); }
    void flushSuperPage(bool keepLastPage = false);
    void fillEmptyHBHs(const IR& ir);

    // expand buffer by positive increment and return old size
    size_t expandBufferBy(size_t by)
    {
      auto offs = buffer.size();
      buffer.resize(offs + by);
      return offs;
    }

    // append to the end of the buffer and return the point where appended to
    size_t pushBack(const char* ptr, size_t sz, bool keepLastOnFlash = true)
    {
      if (!sz) {
        return buffer.size();
      }
      nBytesWritten += sz;
      // do we have a space one this superpage?
      if ((writer->mSuperPageSize - int(buffer.size())) < 0) { // need to flush
        flushSuperPage(keepLastOnFlash);
      }
      auto offs = expandBufferBy(sz);
      memmove(&buffer[offs], ptr, sz);
      return offs;
    }
    // add RDH to buffer. In case this requires flushing of the superpage
    // do not keep the previous page
    size_t pushBack(const RDH& rdh)
    {
      nRDHWritten++;
      return pushBack(reinterpret_cast<const char*>(&rdh), sizeof(RDH), false);
    }

   private:
    static std::vector<char> sCarryOverTrailer;        // working space for optional carry-over trailer
    static std::vector<char> sCarryOverHeader;         // working space for optional carry-over header
    static std::vector<char> sEmptyHBFFiller;          // working space for optional empty HBF filler
    static std::vector<o2::InteractionRecord> sIRWork; // woking buffer for the generated IRs
  };
  //=====================================================================================

  RawFileWriter() = default;
  ~RawFileWriter();
  void close();

  void registerLink(uint16_t fee, uint16_t cru, uint8_t link, uint8_t endpoint, const std::string& outFileName);
  void registerLink(const RDH& rdh, const std::string& outFileName);

  LinkData& getLinkWithSubSpec(LinkSubSpec_t ss) { return mSSpec2Link[ss]; }
  LinkData& getLinkWithSubSpec(const RDH& rdh) { return mSSpec2Link[HBFUtils::getSubSpec(rdh.cruID, rdh.linkID, rdh.endPointID)]; }

  void addData(uint16_t cru, uint8_t lnk, uint8_t endpoint, const IR& ir, const gsl::span<char> data);
  void addData(const RDH& rdh, const IR& ir, const gsl::span<char> data) { addData(rdh.cruID, rdh.linkID, rdh.endPointID, ir, data); }

  void setContinuousReadout() { mROMode = Continuous; }
  void setTriggeredReadout() { mROMode = Triggered; }
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

  int getSuperPageSize() const { return mSuperPageSize; }
  void setSuperPageSize(int nbytes)
  {
    mSuperPageSize = nbytes < 16 * HBFUtils::MAXCRUPage ? HBFUtils::MAXCRUPage : nbytes;
  }

  HBFUtils& getHBFUtils() { return mHBFUtils; }

  template <class T>
  void setCarryOverCallBack(const T* t)
  {
    carryOverFunc = [=](const RDH& rdh, const gsl::span<char> data, const char* ptr, int size, int splitID,
                        std::vector<char>& trailer, std::vector<char>& header) -> int {
      return t->carryOverMethod(rdh, data, ptr, size, splitID, trailer, header);
    };
  }

  template <class T>
  void setEmptyPageCallBack(const T* t)
  {
    emptyHBFFunc = [=](const RDH& rdh, std::vector<char>& toAdd) {
      t->emptyHBFMethod(rdh, toAdd);
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

  int carryOverMethod(const RDH& rdh, const gsl::span<char> data, const char* ptr, int maxSize, int splitID,
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
  void emptyHBFMethod(const RDH& rdh, std::vector<char>& toAdd) const
  {
  }

 private:
  enum RoMode_t { NotSet,
                  Continuous,
                  Triggered };

  IR mIRMax{0, 0}; // highest IR seen
  HBFUtils mHBFUtils;
  std::unordered_map<LinkSubSpec_t, LinkData> mSSpec2Link; // mapping from subSpec to link
  std::unordered_map<std::string, FILE*> mFName2File;      // mapping from filenames to actual files

  CarryOverCallBack carryOverFunc = nullptr; // default call back for large payload splitting (does nothing)
  EmptyPageCallBack emptyHBFFunc = nullptr;  // default call back for empty HBF (does nothing)

  // options
  int mVerbosity = 0;
  int mSuperPageSize = 1024 * 1024; // super page size
  bool mStartTFOnNewSPage = true;   // every TF must start on a new SPage
  RoMode_t mROMode = NotSet;

  ClassDefNV(RawFileWriter, 1);
}; // namespace raw

} // namespace raw
} // namespace o2

#endif //DETECTOR_BASE_RAWFILEWRITER_H
