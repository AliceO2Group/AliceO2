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

#ifndef DETECTOR_BASE_RAWFILEREADER_H
#define DETECTOR_BASE_RAWFILEREADER_H

/// @file   RawFileReader.h
/// @author ruben.shahoyan@cern.ch
/// @brief  Reader for (multiple) raw data files

#include <cstdio>
#include <unordered_map>
#include <map>
#include <tuple>
#include <vector>
#include <string>
#include <utility>
#include <Rtypes.h>
#include "Headers/RAWDataHeader.h"
#include "Headers/DataHeader.h"
#include "DetectorsRaw/RDHUtils.h"

namespace o2
{
namespace raw
{

using IR = o2::InteractionRecord;

struct ReaderInp {
  std::string inifile{};
  std::string rawChannelConfig{};
  std::string dropTF{};
  std::string metricChannel{};
  size_t spSize = 1024L * 1024L;
  size_t bufferSize = 1024L * 1024L;
  size_t minSHM = 0;
  int loop = 1;
  int runNumber = 0;
  uint32_t delay_us = 0;
  uint32_t errMap = 0xffffffff;
  uint32_t minTF = 0;
  uint32_t maxTF = 0xffffffff;
  int verbosity = 0;
  bool partPerSP = true;
  bool cache = false;
  bool autodetectTF0 = false;
  bool preferCalcTF = false;
  bool sup0xccdb = false;
};

class RawFileReader
{
  using LinkSpec_t = uint64_t; // = (origin<<32) | LinkSubSpec
 public:
  //================================================================================
  enum ReadoutCardType { CRU,
                         RORC };
  static constexpr std::string_view CardNames[] = {"CRU", "RORC"};
  //================================================================================
  enum ErrTypes { ErrWrongPacketCounterIncrement,
                  ErrWrongPageCounterIncrement,
                  ErrHBFStopOnFirstPage,
                  ErrHBFNoStop,
                  ErrWrongFirstPage,
                  ErrWrongHBFsPerTF,
                  ErrWrongNumberOfTF,
                  ErrHBFJump,
                  ErrNoSuperPageForTF,
                  ErrNoSOX,
                  ErrMismatchTF,
                  NErrorsDefined
  };

  enum class FirstTFDetection : int { Disabled,
                                      Pending,
                                      Done };

  static constexpr std::string_view ErrNames[] = {
    // long names for error codes
    "Wrong RDH.packetCounter increment",                   // ErrWrongPacketCounterIncrement
    "Wrong RDH.pageCnt increment",                         // ErrWrongPageCounterIncrement
    "RDH.stop set of 1st HBF page",                        // ErrHBFStopOnFirstPage
    "New HBF starts w/o closing old one",                  // ErrHBFNoStop
    "Data does not start with TF/HBF",                     // ErrWrongFirstPage
    "Number of HBFs per TF not as expected",               // ErrWrongHBFsPerTF
    "Number of TFs is less than expected",                 // ErrWrongNumberOfTF
    "Wrong HBF orbit increment",                           // ErrHBFJump
    "TF does not start by new superpage",                  // ErrNoSuperPageForTF
    "No SOX found on 1st page",                            // ErrNoSOX
    "Mismatch between flagged and calculated new TF start" // ErrMismatchTF
  };
  static constexpr std::string_view ErrNamesShort[] = {
    // short names for error codes
    "packet-increment", // ErrWrongPacketCounterIncrement
    "page-increment",   // ErrWrongPageCounterIncrement
    "stop-on-page0",    // ErrHBFStopOnFirstPage
    "missing-stop",     // ErrHBFNoStop
    "starts-with-tf",   // ErrWrongFirstPage
    "hbf-per-tf",       // ErrWrongHBFsPerTF
    "tf-per-link",      // ErrWrongNumberOfTF
    "hbf-jump",         // ErrHBFJump
    "no-spage-for-tf",  // ErrNoSuperPageForTF
    "no-sox",           // ErrNoSOX
    "tf-start-mismatch" // ErrMismatchTF
  };
  static constexpr bool ErrCheckDefaults[] = {
    true,  // ErrWrongPacketCounterIncrement
    true,  // ErrWrongPageCounterIncrement
    false, // ErrHBFStopOnFirstPage
    true,  // ErrHBFNoStop
    true,  // ErrWrongFirstPage
    true,  // ErrWrongHBFsPerTF
    true,  // ErrWrongNumberOfTF
    true,  // ErrHBFJump
    false, // ErrNoSuperPageForTF
    false, // ErrNoSOX
    true,  // ErrMismatchTF
  };
  //================================================================================

  using RDHAny = header::RDHAny;
  using RDH = o2::header::RAWDataHeader;
  using OrigDescCard = std::tuple<o2::header::DataOrigin, o2::header::DataDescription, ReadoutCardType>;
  using InputsMap = std::map<OrigDescCard, std::vector<std::string>>;

  //=====================================================================================

  // reference on blocks making single message part
  struct PartStat {
    int size;    // total size
    int nBlocks; // number of consecutive LinkBlock objects
  };

  // info on the smallest block of data to be read when fetching the HBF
  struct LinkBlock {
    enum { StartTF = 0x1,
           StartHB = 0x1 << 1,
           StartSP = 0x1 << 2,
           EndHB = 0x1 << 3 };
    size_t offset = 0;                 //! where data of the block starts
    uint32_t size = 0;                 //! block size
    uint32_t tfID = 0;                 //! tf counter (from 0)
    IR ir = 0;                         //! ir starting the block
    uint16_t fileID = 0;               //! file id where the block is located
    uint8_t flags = 0;                 //! different flags
    std::unique_ptr<char[]> dataCache; //! optional cache for fast access
    LinkBlock() = default;
    LinkBlock(int fid, size_t offs) : offset(offs), fileID(fid) {}
    void setFlag(uint8_t fl, bool v = true)
    {
      if (v) {
        flags |= fl;
      } else {
        flags &= ~fl;
      }
    }
    bool testFlag(uint8_t fl) const { return (flags & fl) == fl; }
    void print(const std::string& pref = "") const;
  };

  //=====================================================================================
  struct LinkData {
    RDHAny rdhl; //! RDH with the running info of the last RDH seen
    o2::InteractionRecord irOfSOX{};
    LinkSpec_t spec = 0;       //! Link subspec augmented by its origin
    LinkSubSpec_t subspec = 0; //! subspec according to DataDistribution
    uint32_t nTimeFrames = 0;  //!
    uint32_t nHBFrames = 0;    //!
    uint32_t nSPages = 0;      //!
    uint64_t nCRUPages = 0;    //!
    bool cruDetector = true;   //! CRU vs RORC detector
    bool continuousRO = true;  //!

    o2::header::DataOrigin origin = o2::header::gDataOriginInvalid;                //!
    o2::header::DataDescription description = o2::header::gDataDescriptionInvalid; //!
    int nErrors = 0;                                                               //!
    std::vector<LinkBlock> blocks;                                                 //!
    std::vector<std::pair<int, uint32_t>> tfStartBlock;
    //
    // transient info during processing
    bool openHB = false;    //!
    int nHBFinTF = 0;       //!
    int nextBlock2Read = 0; //! next block which should be read

    LinkData() = default;
    template <typename H>
    LinkData(const H& rdh, RawFileReader* r) : rdhl(rdh), reader(r)
    {
    }
    bool preprocessCRUPage(const RDHAny& rdh, bool newSPage);
    size_t getLargestSuperPage() const;
    size_t getLargestTF() const;
    size_t getNextHBFSize() const;
    size_t getNextTFSize() const;
    size_t getNextTFSuperPagesStat(std::vector<PartStat>& parts) const;
    int getNHBFinTF() const;

    size_t readNextHBF(char* buff);
    size_t readNextTF(char* buff);
    size_t readNextSuperPage(char* buff, const PartStat* pstat = nullptr);
    size_t skipNextHBF();
    size_t skipNextTF();

    bool rewindToTF(uint32_t tf);
    void print(bool verbose = false, const std::string& pref = "") const;
    std::string describe() const;

   private:
    RawFileReader* reader = nullptr; //!
  };

  //=====================================================================================

  RawFileReader(const std::string& config = "", int verbosity = 0, size_t buffsize = 50 * 1024UL);
  ~RawFileReader() { clear(); }

  void loadFromInputsMap(const InputsMap& inp);
  bool init();
  void clear();
  bool addFile(const std::string& sname, o2::header::DataOrigin origin, o2::header::DataDescription desc, ReadoutCardType t = CRU);
  bool addFile(const std::string& sname) { return addFile(sname, mDefDataOrigin, mDefDataDescription, mDefCardType); }
  void setDefaultReadoutCardType(ReadoutCardType t = CRU) { mDefCardType = t; }
  void setDefaultDataOrigin(const std::string& orig) { mDefDataOrigin = getDataOrigin(orig); }
  void setDefaultDataDescription(const std::string& desc) { mDefDataDescription = getDataDescription(desc); }
  void setDefaultDataOrigin(const o2::header::DataOrigin o) { mDefDataOrigin = o; }
  void setDefaultDataDescription(const o2::header::DataDescription d) { mDefDataDescription = d; }
  int getNLinks() const { return mLinksData.size(); }
  int getNFiles() const { return mFiles.size(); }

  uint32_t getNextTFToRead() const { return mNextTF2Read; }
  void setNextTFToRead(uint32_t tf) { mNextTF2Read = tf; }

  const std::vector<int>& getLinksOrder() const { return mOrderedIDs; }
  const LinkData& getLink(int i) const { return mLinksData[mOrderedIDs[i]]; }
  const LinkData& getLinkWithSpec(LinkSpec_t s) const { return mLinksData[mLinkEntries.at(s)]; }
  LinkData& getLink(int i) { return mLinksData[mOrderedIDs[i]]; }
  LinkSubSpec_t getLinkSubSpec(int i) const { return getLink(i).subspec; }
  LinkSpec_t getLinkSpec(int i) const { return getLink(i).spec; }

  void printStat(bool verbose = false) const;

  void setVerbosity(int v = 1) { mVerbosity = v; }
  void setCheckErrors(uint32_t m = 0xffffffff) { mCheckErrors = m & ((0x1 << NErrorsDefined) - 1); }
  int getVerbosity() const { return mVerbosity; }
  uint32_t getCheckErrors() const { return mCheckErrors; }
  bool isProcessingStopped() const { return mStopProcessing; }

  void setNominalSPageSize(int n = 0x1 << 20) { mNominalSPageSize = n > (0x1 << 15) ? n : (0x1 << 15); }
  int getNominalSPageSize() const { return mNominalSPageSize; }

  void setBufferSize(size_t s) { mBufferSize = s < sizeof(RDHAny) ? sizeof(RDHAny) : s; }
  size_t getBufferSize() const { return mBufferSize; }

  void setMaxTFToRead(uint32_t n) { mMaxTFToRead = n; }
  bool isEmpty() const { return mEmpty; }
  uint32_t getMaxTFToRead() const { return mMaxTFToRead; }
  uint32_t getNTimeFrames() const { return mNTimeFrames; }
  uint32_t getOrbitMin() const { return mOrbitMin; }
  uint32_t getOrbitMax() const { return mOrbitMax; }

  bool getCacheData() const { return mCacheData; }
  void setCacheData(bool v) { mCacheData = v; }

  o2::header::DataOrigin getDefaultDataOrigin() const { return mDefDataOrigin; }
  o2::header::DataDescription getDefaultDataSpecification() const { return mDefDataDescription; }
  ReadoutCardType getDefaultReadoutCardType() const { return mDefCardType; }

  void imposeFirstTF(uint32_t orbit);
  void setTFAutodetect(FirstTFDetection v) { mFirstTFAutodetect = v; }
  void setPreferCalculatedTFStart(bool v) { mPreferCalculatedTFStart = v; }
  FirstTFDetection getTFAutodetect() const { return mFirstTFAutodetect; }

  void setIROfSOX(const o2::InteractionRecord& ir);

  static o2::header::DataOrigin getDataOrigin(const std::string& ors);
  static o2::header::DataDescription getDataDescription(const std::string& ors);
  static InputsMap parseInput(const std::string& confUri);
  static std::string nochk_opt(ErrTypes e);
  static std::string nochk_expl(ErrTypes e);

 private:
  int getLinkLocalID(const RDHAny& rdh, int fileID);
  bool preprocessFile(int ifl);
  static LinkSpec_t createSpec(o2::header::DataOrigin orig, LinkSubSpec_t ss) { return (LinkSpec_t(orig) << 32) | ss; }

  static constexpr o2::header::DataOrigin DEFDataOrigin = o2::header::gDataOriginFLP;
  static constexpr o2::header::DataDescription DEFDataDescription = o2::header::gDataDescriptionRawData;
  static constexpr ReadoutCardType DEFCardType = CRU;
  o2::header::DataOrigin mDefDataOrigin = DEFDataOrigin;                //!
  o2::header::DataDescription mDefDataDescription = DEFDataDescription; //!
  ReadoutCardType mDefCardType = CRU;                                   //!
  std::vector<std::string> mFileNames;                                  //! input file names
  std::vector<FILE*> mFiles;                                            //! input file handlers
  std::vector<std::unique_ptr<char[]>> mFileBuffers;                    //! buffers for input files
  std::vector<OrigDescCard> mDataSpecs;                                 //! data origin and description for every input file + readout card type
  bool mInitDone = false;
  bool mEmpty = true;
  std::unordered_map<LinkSpec_t, int> mLinkEntries;                 //! mapping between RDH specs and link entry in the mLinksData
  std::vector<LinkData> mLinksData;                                 //! info on links data in the files
  std::vector<int> mOrderedIDs;                                     //! links entries ordered in Specs
  uint32_t mMaxTFToRead = 0xffffffff;                               //! max TFs to process
  uint32_t mNTimeFrames = 0;                                        //! total number of time frames
  uint32_t mNextTF2Read = 0;                                        //! next TF to read
  uint32_t mOrbitMin = 0xffffffff;                                  //! lowest orbit seen by any link
  uint32_t mOrbitMax = 0;                                           //! highest orbit seen by any link
  size_t mBufferSize = 5 * 1024UL;                                  //! size of the buffer for files reading
  int mNominalSPageSize = 0x1 << 20;                                //! expected super-page size in B
  int mCurrentFileID = 0;                                           //! current file being processed
  long int mPosInFile = 0;                                          //! current position in the file
  bool mMultiLinkFile = false;                                      //! was > than 1 link seen in the file?
  bool mCacheData = false;                                          //! cache data to block after 1st scan (may require excessive memory, use with care)
  bool mStopProcessing = false;                                     //! stop processing after error
  uint32_t mCheckErrors = 0;                                        //! mask for errors to check
  FirstTFDetection mFirstTFAutodetect = FirstTFDetection::Disabled; //!
  bool mPreferCalculatedTFStart = false;                            //! prefer TFstart calculated via HBFUtils
  int mVerbosity = 0;                                               //!
  ClassDefNV(RawFileReader, 1);
};

} // namespace raw
} // namespace o2

#endif //DETECTOR_BASE_RAWFILEREADER_H
