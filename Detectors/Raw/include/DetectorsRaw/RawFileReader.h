// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include <vector>
#include <string>
#include <utility>
#include <Rtypes.h>
#include "Headers/RAWDataHeader.h"
#include "Headers/DataHeader.h"
#include "DetectorsRaw/HBFUtils.h"

namespace o2
{
namespace raw
{

class RawFileReader
{
  using LinkSpec_t = uint64_t; // = (origin<<32) | LinkSubSpec
 public:
  using RDH = o2::header::RAWDataHeaderV4;
  using OrDesc = std::pair<o2::header::DataOrigin, o2::header::DataDescription>;
  using InputsMap = std::map<OrDesc, std::vector<std::string>>;
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
                  NErrorsDefined
  };
  static constexpr std::string_view ErrNames[] = {
    // long names for error codes
    "Wrong RDH.packetCounter increment",     // ErrWrongPacketCounterIncrement
    "Wrong RDH.pageCnt increment",           // ErrWrongPageCounterIncrement
    "RDH.stop set of 1st HBF page",          // ErrHBFStopOnFirstPage
    "New HBF starts w/o closing old one",    // ErrHBFNoStop
    "Data does not start with TF/HBF",       // ErrWrongFirstPage
    "Number of HBFs per TF not as expected", // ErrWrongHBFsPerTF
    "Number of TFs is less than expected",   // ErrWrongNumberOfTF
    "Wrong HBF orbit increment",             // ErrHBFJump
    "TF does not start by new superpage"     // ErrNoSuperPageForTF
  };
  static constexpr std::string_view ErrNamesShort[] = { // short names for error codes
    "packet-increment",
    "page-increment",
    "stop-on-page0",
    "missing-stop",
    "starts-with-tf",
    "hbf-per-tf",
    "tf-per-link",
    "hbf-jump",
    "no-spage-for-tf"};
  //================================================================================

  //=====================================================================================
  // info on the smallest block of data to be read when fetching the HBF
  struct LinkBlock {
    enum { StartTF = 0x1,
           StartHB = 0x1 << 1,
           StartSP = 0x1 << 2,
           EndHB = 0x1 << 3 };
    size_t offset = 0;    // where data of the block starts
    uint32_t size = 0;    // block size
    uint32_t tfID = 0;    // tf counter (from 0)
    uint32_t orbit = 0;   // orbit starting the block
    uint16_t fileID = 0;  // file id where the block is located
    uint8_t flags = 0;    // different flags
    LinkBlock() = default;
    LinkBlock(int fid, size_t offs) : offset(offs), fileID(fid) {}
    void setFlag(uint8_t fl, bool v = true)
    {
      if (v)
        flags |= fl;
      else
        flags &= ~fl;
    }
    bool testFlag(uint8_t fl) const { return (flags & fl) == fl; }
    void print(const std::string& pref = "") const;
  };

  //=====================================================================================
  struct LinkData {
    RDH rdhl;             // RDH with the running info of the last RDH seen
    LinkSpec_t spec = 0;  // Link subspec augmented by its origin
    LinkSubSpec_t subspec = 0; // subspec according to DataDistribution
    uint32_t nTimeFrames = 0;
    uint32_t nHBFrames = 0;
    uint32_t nCRUPages = 0;
    uint32_t nSPages = 0;
    o2::header::DataOrigin origin = o2::header::gDataOriginInvalid;
    o2::header::DataDescription description = o2::header::gDataDescriptionInvalid;
    std::string fairMQChannel{}; // name of the fairMQ channel for the output
    int nErrors = 0;
    std::vector<LinkBlock> blocks;
    //
    // transient info during processing
    bool openHB = false;
    int nHBFinTF = 0;
    int nextBlock2Read = 0; // next block which should be read

    LinkData() = default;
    LinkData(const o2::header::RAWDataHeaderV4& rdh, const RawFileReader* r);
    LinkData(const o2::header::RAWDataHeaderV5& rdh, const RawFileReader* r);
    bool preprocessCRUPage(const RDH& rdh, bool newSPage);
    size_t getLargestSuperPage() const;
    size_t getLargestTF() const;
    size_t getNextHBFSize() const;
    size_t getNextTFSize() const;
    int getNHBFinTF() const;

    size_t readNextHBF(char* buff);
    size_t readNextTF(char* buff);

    void print(bool verbose = false, const std::string& pref = "") const;
    std::string describe() const;

   private:
    const RawFileReader* reader = nullptr;
  };

  //=====================================================================================

  RawFileReader(const std::string& config = "", int verbosity = 0);
  ~RawFileReader() { clear(); }

  void loadFromInputsMap(const InputsMap& inp);
  bool init();
  void clear();
  bool addFile(const std::string& sname, o2::header::DataOrigin origin, o2::header::DataDescription desc);
  bool addFile(const std::string& sname) { return addFile(sname, mDefDataOrigin, mDefDataDescription); }
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

  void setNominalSPageSize(int n = 0x1 << 20) { mNominalSPageSize = n > (0x1 << 15) ? n : (0x1 << 15); }
  int getNominalSPageSize() const { return mNominalSPageSize; }

  void setNominalHBFperTF(int n = 256) { mNominalHBFperTF = n > 1 ? n : 1; }
  int getNominalHBFperTF() const { return mNominalHBFperTF; }

  uint32_t getNTimeFrames() const { return mNTimeFrames; }
  uint32_t getOrbitMin() const { return mOrbitMin; }
  uint32_t getOrbitMax() const { return mOrbitMax; }

  o2::header::DataOrigin getDefaultDataOrigin() const { return mDefDataOrigin; }
  o2::header::DataDescription getDefaultDataSpecification() const { return mDefDataDescription; }

  static o2::header::DataOrigin getDataOrigin(const std::string& ors);
  static o2::header::DataDescription getDataDescription(const std::string& ors);
  static InputsMap parseInput(const std::string& confUri);

 private:
  int getLinkLocalID(const RDH& rdh, o2::header::DataOrigin orig);
  bool preprocessFile(int ifl);
  static LinkSpec_t createSpec(o2::header::DataOrigin orig, LinkSubSpec_t ss) { return (LinkSpec_t(orig) << 32) | ss; }

  static constexpr o2::header::DataOrigin DEFDataOrigin = o2::header::gDataOriginFLP;
  static constexpr o2::header::DataDescription DEFDataDescription = o2::header::gDataDescriptionRawData;

  o2::header::DataOrigin mDefDataOrigin = DEFDataOrigin;
  o2::header::DataDescription mDefDataDescription = DEFDataDescription;

  std::vector<std::string> mFileNames; // input file names
  std::vector<FILE*> mFiles;           // input file handlers
  std::vector<OrDesc> mDataSpecs;      // data origin and description for every input file
  bool mInitDone = false;
  std::unordered_map<LinkSpec_t, int> mLinkEntries; // mapping between RDH specs and link entry in the mLinksData
  std::vector<LinkData> mLinksData;                 // info on links data in the files
  std::vector<int> mOrderedIDs;                     // links entries ordered in Specs
  uint32_t mNTimeFrames = 0;                        // total number of time frames
  uint32_t mNextTF2Read = 0;                        // next TF to read
  uint32_t mOrbitMin = 0xffffffff;                  // lowest orbit seen by any link
  uint32_t mOrbitMax = 0;                           // highest orbit seen by any link
  int mNominalSPageSize = 0x1 << 20;                // expected super-page size in B
  int mNominalHBFperTF = 256;                       // expected N HBF per TF
  int mCurrentFileID = 0;                           // current file being processed
  long int mPosInFile = 0;                          // current position in the file
  bool mMultiLinkFile = false;                      // was > than 1 link seen in the file?
  uint32_t mCheckErrors = 0;                        // mask for errors to check
  int mVerbosity = 0;

  ClassDefNV(RawFileReader, 1);
};


} // namespace raw
} // namespace o2

#endif //DETECTOR_BASE_RAWFILEREADER_H
