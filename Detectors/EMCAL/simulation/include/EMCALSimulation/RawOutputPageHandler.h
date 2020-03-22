// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <exception>
#include <map>
#include <string>
#include <vector>
#include <CommonDataFormat/InteractionRecord.h>
#include <EMCALSimulation/DMAOutputStream.h>

namespace o2
{
namespace emcal
{

/// \class RawOutputPageHandler
/// \brief Handler for EMCAL raw page buffering, timeframe building and output streaming
/// \ingroup EMCALsimulation
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \since March 22, 2020
///
/// # General
/// The raw format for EMCAL consists of separate superpages of timeframes arranged per
/// (DDL) link. Each timeframe consists of
/// - Empty RDH without payload indicating start of timeframe. Counters are reset
/// - Pages with RDH and payload for each trigger BC within the timeframe
/// - Empty RDH without payload indicating end of timeframe. Contains the number of pages
///   in the timeframe, the stop bit and the timeframe trigger.
/// At the start of data an empty timeframe (without pages) consisting only of star and stop
/// RDH will mark the beginning of the data stream.
///
/// # Usage
/// For each new trigger the function initTrigger is to be called with the InteractionRecord
/// of the bunch crossing corresponding to the trigger. Pages created by the raw writer and added
/// via addPageForLink are buffered per link. The HBFUtils, provided from outside, decide on whether
/// a new timeframe is triggered. In case of a new timeframe raw timeframes are created for all
/// links with pages, containing all pages of a link and streamed to file (in initTrigger). In case
/// the buffer still contains pages when the destructor is called, the pages are streamed as remaining
/// pages of the last timeframe (in the destructor).
class RawOutputPageHandler
{
 public:
  using RawHeader = o2::header::RAWDataHeaderV4;

  /// \class LinkIDException
  /// \brief Exception handling invalid link IDs (outside the range of EMCAL links)
  class LinkIDException : public std::exception
  {
   public:
    /// \brief Constructor, defining
    LinkIDException(int linkID);

    /// \brief Destructor
    ~LinkIDException() noexcept final = default;

    /// \brief Access to error message of the exception
    /// \return Error message of the exception
    const char* what() const noexcept final { return mMessage.data(); }

    /// \brief Access to Link ID raising the exception
    /// \return Link ID raising the exception
    int getLinkID() const { return mLinkID; }

   private:
    int mLinkID;          ///< ID of the link raising the exception
    std::string mMessage; ///< Error message;
  };

  /// \class RawPageBuffer
  /// \brief Buffer for raw pages
  ///
  /// Internal helper class for buffering raw pages within a timeframe for a certain link.
  /// The functionality consists of:
  /// - add page: adding new page to the raw buffer
  /// - getPages: access to all pages. Mainly used when building the timeframe
  /// - flush: clear buffer (after timeframe building)
  class RawPageBuffer
  {
   public:
    /// \struct PageData
    /// \brief Structure for a raw page (header and payload words)
    struct PageData {
      RawHeader mHeader;       ///< Header template of the page, containing link ID, fee ID and trigger
      std::vector<char> mPage; ///< Payload page
    };

    /// \brief Constructor
    RawPageBuffer() = default;

    /// \brief Destructor
    ~RawPageBuffer() = default;

    /// \brief Adding new raw page to the page buffer
    /// \brief header Template raw header, containing link/fee ID and trigger
    /// \brief page Raw page (as char words) to be added to the buffer
    void addPage(const RawHeader& header, const std::vector<char>& page) { mPages.push_back({header, page}); }

    /// \brief Cleaning page buffer
    void flush() { mPages.clear(); }

    /// \brief Access to pages in the buffer
    /// \return vector with all pages in the buffer
    const std::vector<PageData>& getPages() const { return mPages; }

   private:
    std::vector<PageData> mPages; ///< Buffer for pages
  };

  /// \brief Constructor, initializing page handler with output filename and HBF utils
  /// \param rawfilename Name of the output file
  RawOutputPageHandler(const char* rawfilename);

  /// \brief Destructor
  ///
  /// In case buffers contain payload the page is streamed as last
  /// timeframe for the links containing payload pages.
  ~RawOutputPageHandler();

  /// \brief Initialize new trigger
  /// \param currentIR Interaction record of the collision triggering
  ///
  /// Initialize new trigger. In case the Timeframe changes with the trigger
  /// the buffers belonging to the previous buffer are streamed to file. For
  /// each link a separate timeframe is created, starting with empty open/close
  /// raw data header. Timeframes are only created for links which buffer pages.
  void initTrigger(const o2::InteractionRecord& currentIR);

  /// \brief Add new page for link to the page buffer
  /// \param linkID ID of the link
  /// \param header Template raw header of the page
  /// \param dmapage Payload page
  /// \throw LinkIDException in case link ID is invalid
  void addPageForLink(int linkID, const RawHeader& header, const std::vector<char>& dmapage);

 private:
  /// \brief Write timeframe for an entire link
  /// \param linkID ID of the link
  /// \param pagebuffer Buffer with raw pages in timeframe belonging to the link
  ///
  /// Timeframes for raw data in EMCAL contain:
  /// - Empty RDH, indicating start of timeframe. No payload assigned. Counters 0
  /// - Raw page for every trigger in the timeframe: Each raw page starts with a
  ///   RDH. In case the payload exceeds the 8 kB page it is split into multiple
  ///   pages. The page counter is calculated with respect to the first header in a
  ///   timeframe
  /// - Empty RDH, closing the timeframe, containing the number of pages in the
  ///   timeframe
  void writeTimeframe(int linkID, const RawPageBuffer& pagebuffer);

  int mCurrentTimeframe = -1;             ///< Current timeframe ID (needed to detect whether a new timeframe starts)
  std::map<int, RawPageBuffer> mRawPages; ///< Buffer with raw pages for all links
  DMAOutputStream mOutputStream;          ///< File output stream
};
} // namespace emcal
} // namespace o2