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
#include <vector>
#include <string>
#include <Rtypes.h>
#include "Headers/RAWDataHeader.h"

namespace o2
{
namespace raw
{

class RawFileReader
{

 public:
  using RDH = o2::header::RAWDataHeaderV4;
  using LinkSpec_t = uint32_t;
  //================================================================================
  enum ErrTypes { ErrWrongPacketCounterIncrement,
                  ErrWrongPageCounterIncrement,
                  ErrHBFStopOnFirstPage,
                  ErrHBFNoStop,
                  ErrWrongFirstPage,
                  ErrWrongHBFsPerTF,
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
    "hbf-jump",
    "no-spage-for-tf"};
  //================================================================================

  //=====================================================================================
  struct LinkBlock {
    size_t offset = 0;    // where data of the block starts
    uint32_t size = 0;    // block size
    uint32_t orbit = 0;   // orbit starting the block
    uint16_t fileID = 0;  // file id where the file is located
    bool startTF = false; // does this block starts a new TF ?
    bool startHB = false; // does this block starts a new HBF ?
    bool startSP = false; // does the block correspond to new superpage?
    bool endHB = false;   // ends by RDH with stop (HBF close)
    LinkBlock() = default;
    LinkBlock(int fid, size_t offs) : offset(offs), fileID(fid) {}
    void print(const std::string pref = "") const;
  };

  //=====================================================================================
  struct LinkData {
    RDH rdhl;             // RDH with the running info of the last RDH seen
    uint32_t subspec = 0; // subspec according to DataDistribution
    uint32_t nTimeFrames = 0;
    uint32_t nHBFrames = 0;
    uint32_t nCRUPages = 0;
    uint32_t nSPages = 0;
    int nErrors = 0;
    //
    // transient info during pre-processing
    bool openHB = false;
    int nHBFinTF = 0;
    std::vector<LinkBlock> blocks;

    LinkData() = default;
    LinkData(const o2::header::RAWDataHeaderV4& rdh, const RawFileReader* r);
    LinkData(const o2::header::RAWDataHeaderV5& rdh, const RawFileReader* r);
    bool preprocessCRUPage(const RDH& rdh, bool newSPage);
    size_t getLargestSuperPage() const;
    size_t getLargestTF() const;
    void print(bool verbose = false, const std::string pref = "") const;

   private:
    const RawFileReader* reader = nullptr;
  };

  //=====================================================================================

  ~RawFileReader() { clear(); }
  bool init();
  void clear();
  bool addFile(const std::string& sname);
  int getNLinks() const { return mLinksData.size(); }
  int getNFiles() const { return mFiles.size(); }

  const std::vector<int>& getLinksOrder() const { return mOrderedIDs; }
  const LinkData& getLink(int i) const { return mLinksData[mOrderedIDs[i]]; }
  const LinkData& getLinkWithSubSpec(LinkSpec_t s) const { return mLinksData[mLinkEntries.at(s)]; }
  int getLinkSubSpec(int i) const { return getLink(i).subspec; }

  void printStat(bool verbose = false) const;

  void setVerbosity(int v = 1) { mVerbosity = v; }
  void setCheckErrors(uint32_t m = 0xffffffff) { mCheckErrors = m & ((0x1 << NErrorsDefined) - 1); }
  int getVerbosity() const { return mVerbosity; }
  uint32_t getCheckErrors() const { return mCheckErrors; }

  void setNominalSPageSize(int n = 0x1 << 20) { mNominalSPageSize = n > (0x1 << 15) ? n : (0x1 << 15); }
  int getNominalSPageSize() const { return mNominalSPageSize; }

  void setNominalHBFperTF(int n = 256) { mNominalHBFperTF = n > 1 ? n : 1; }
  int getNominalHBFperTF() const { return mNominalHBFperTF; }

 private:
  bool checkRDH(const o2::header::RAWDataHeaderV4& rdh) const;
  bool checkRDH(const o2::header::RAWDataHeaderV5& rdh) const;
  LinkSpec_t getSubSpec(uint16_t cru, uint8_t link, uint8_t endpoint) const;
  int getLinkLocalID(const RDH& rdh);
  bool preprocessFile(int ifl);

  std::vector<std::string> mFileNames; // input file names
  std::vector<FILE*> mFiles;           // input file handlers
  bool mInitDone = false;
  std::unordered_map<LinkSpec_t, int> mLinkEntries; // mapping between RDH specs and link entry in the mLinksData
  std::vector<LinkData> mLinksData;                 // info on links data in the files
  std::vector<int> mOrderedIDs;                     // links entries ordered in Specs
  int mNominalSPageSize = 0x1 << 20;                // expected super-page size in B
  int mNominalHBFperTF = 256;                       // expected N HBF per TF
  int mCurrentFileID = 0;                           // current file being processed
  long int mPosInFile = 0;                          // current position in the file
  bool mMultiLinkFile = false;                      // was > than 1 link seen in the file?
  uint32_t mCheckErrors = 0;                        // mask for errors to check
  int mVerbosity = 0;

  ClassDefNV(RawFileReader, 1);
};

//_____________________________________________________________________
inline RawFileReader::LinkSpec_t RawFileReader::getSubSpec(uint16_t cru, uint8_t link, uint8_t endpoint) const
{
  // define subspecification as in DataDistribution
  int linkValue = (RawFileReader::LinkSpec_t(link) + 1) << (endpoint == 1 ? 8 : 0);
  return (RawFileReader::LinkSpec_t(cru) << 16) | linkValue;
}

} // namespace raw
} // namespace o2

#endif //DETECTOR_BASE_RAWFILEREADER_H
