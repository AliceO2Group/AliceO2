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
#ifndef FRAMEWORK_UTILS_DPLRAWPARSER_H
#define FRAMEWORK_UTILS_DPLRAWPARSER_H

/// @file   DPLRawParser.h
/// @author Matthias Richter
/// @since  2020-02-27
/// @brief  A raw page parser for DPL input

#include "DPLUtils/RawParser.h"
#include "Framework/InputRecord.h"
#include "Framework/DataRef.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessingHeader.h"
#include "Headers/DataHeader.h"
#include <utility> // std::declval

namespace o2::framework
{

/// @class DPLRawParser
/// @brief The parser handles transparently input in the format of raw pages.
///
/// A DPL processor will receive raw pages accumulated on three levels:
///   1) the DPL processor has one or more input route(s)
///   2) multiple parts per input route (split payloads or multiple input
///      specs matching the same route spec
///   3) variable number of raw pages in one payload
///
/// Internally, class @ref RawParser is used to access raw pages withon one
/// payload message and dynamically adopt to RAWDataHeader version.
///
/// The parser provides an iterator interface to loop over raw pages,
/// starting at the first raw page of the first payload at the first route
/// and going to the next route when all payloads are processed. The iterator
/// element is @ref RawDataHeaderInfo containing just the two least significant
/// bytes of the RDH where we have the version and header size.
///
/// The iterator object provides methods to access the concrete RDH, the raw
/// buffer, the payload, etc.
///
/// Usage:
///   DPLRawParser parser(inputs);
///   for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
///     // retrieving RDH v4
///     auto const* rdh = it.get_if<o2::header::RAWDataHeaderV4>();
///     // retrieving the raw pointer of the page
///     auto const* raw = it.raw();
///     // retrieving payload pointer of the page
///     auto const* payload = it.data();
///     // size of payload
///     size_t payloadSize = it.size();
///     // offset of payload in the raw page
///     size_t offset = it.offset();
///   }
template <bool BOUNDS_CHECKS = true>
class DPLRawParser
{
 public:
  using rawparser_type = RawParser<8192, BOUNDS_CHECKS>;
  using buffer_type = typename rawparser_type::buffer_type;

  DPLRawParser() = delete;
  DPLRawParser(InputRecord& inputs, std::vector<InputSpec> filterSpecs = {}, fair::Severity sev = fair::Severity::alarm) : mInputs(inputs), mFilterSpecs(filterSpecs), mSeverity(sev) {}

  void setMaxFailureMessages(size_t n) { mMaxFailureMessages = n; }
  void setExtFailureCounter(size_t* cnt) { mExtFailureCounter = cnt; }
  static void setCheckIncompleteHBF(bool v) { rawparser_type::setCheckIncompleteHBF(v); }

  // this is a dummy default buffer used to initialize the RawParser in the iterator
  // constructor
  static constexpr o2::header::RAWDataHeader initializer = o2::header::RAWDataHeader{.memorySize = sizeof(o2::header::RAWDataHeader)};

  /// Iterator implementation
  /// Supports the following operations:
  /// - increment (there is no decrement, its not a bidirectional parser)
  /// - dereference operator returns @a RawDataHeaderInfo as common header
  /// - member function data() returns pointer to payload at current position
  /// - member function size() return size of payload at current position
  template <typename T>
  class Iterator
  {
   public:
    using iterator_category = std::forward_iterator_tag;
    using self_type = Iterator;
    using value_type = T;
    using reference = T&;
    using pointer = T*;
    // the iterator over the input channels
    using input_iterator = decltype(std::declval<InputRecord>().begin());
    // the parser type
    using parser_type = rawparser_type const;

    Iterator() = delete;

    Iterator(InputRecord& parent, input_iterator it, input_iterator end, std::vector<InputSpec> const& filterSpecs, fair::Severity sev = fair::Severity::alarm, size_t maxErrMsg = -1, size_t* cntErrMsg = nullptr)
      : mParent(parent), mInputIterator(it), mEnd(end), mPartIterator(mInputIterator.begin()), mParser(std::make_unique<parser_type>(reinterpret_cast<const char*>(&initializer), sizeof(initializer))), mCurrent(mParser->begin()), mFilterSpecs(filterSpecs), mMaxFailureMessages(maxErrMsg), mExtFailureCounter(cntErrMsg), mSeverity(sev)
    {
      mParser.reset();
      next();
    }

    ~Iterator() = default;

    // prefix increment
    self_type& operator++()
    {
      next();
      return *this;
    }
    // postfix increment
    self_type operator++(int /*unused*/)
    {
      self_type copy(*this);
      operator++();
      return copy;
    }
    // return reference
    reference operator*()
    {
      return *mCurrent;
    }
    // comparison
    bool operator==(const self_type& other) const
    {
      bool result = mInputIterator == other.mInputIterator;
      result = result && mPartIterator == other.mPartIterator;
      if (mParser != nullptr && other.mParser != nullptr) {
        result = result && mCurrent == other.mCurrent;
      }
      return result;
    }

    bool operator!=(const self_type& rh) const
    {
      return not operator==(rh);
    }

    /// get DataHeader of the current input message
    o2::header::DataHeader const* o2DataHeader() const
    {
      if (mInputIterator != mEnd) {
        return DataRefUtils::getHeader<o2::header::DataHeader*>(*mPartIterator);
      }
      return nullptr;
    }

    /// get DataProcessingHeader of the current input message
    o2::framework::DataProcessingHeader const* o2DataProcessingHeader() const
    {
      if (mInputIterator != mEnd) {
        return DataRefUtils::getHeader<o2::framework::DataProcessingHeader*>(*mPartIterator);
      }
      return nullptr;
    }

    /// get pointer to raw block at current position, rdh starts here
    buffer_type const* raw() const
    {
      return mCurrent.raw();
    }

    /// get pointer to payload at current position
    buffer_type const* data() const
    {
      return mCurrent.data();
    }

    /// offset of payload at current position
    size_t offset() const
    {
      return mCurrent.offset();
    }

    /// get size of payload at current position
    size_t size() const
    {
      return mCurrent.size();
    }

    /// get header as specific type
    /// @return pointer to header of the specified type, or nullptr if type does not match to actual type
    template <typename U>
    U const* get_if() const
    {
      return mCurrent.template get_if<U>();
    }

    friend std::ostream& operator<<(std::ostream& os, self_type const& it)
    {
      if (it.mInputIterator != it.mEnd && it.mPartIterator != it.mInputIterator.end() && it.mParser != nullptr) {
        os << it.mCurrent;
      }
      return os;
    }

    // helper wrapper to control the format and content of the stream output
    template <raw_parser::FormatSpec FmtCtrl>
    struct Fmt {
      static constexpr raw_parser::FormatSpec format_control = FmtCtrl;
      Fmt(self_type const& _it) : it{_it} {}
      self_type const& it;
    };

    template <raw_parser::FormatSpec FmtCtrl>
    friend std::ostream& operator<<(std::ostream& os, Fmt<FmtCtrl> const& fmt)
    {
      auto const& it = fmt.it;
      if (it.mInputIterator != it.mEnd && it.mPartIterator != it.mInputIterator.end() && it.mParser != nullptr) {
        if constexpr (FmtCtrl == raw_parser::FormatSpec::Info) {
          // TODO: need to propagate the format spec also on the RawParser object
          // for now this operation prints the RDH version info and the table header
          os << *it.mParser;
        } else {
          os << it;
        }
      }
      return os;
    }

   private:
    // the iterator over the parts in one channel
    using part_iterator = typename input_iterator::const_iterator;
    // the iterator over the over the parser pages
    using parser_iterator = typename parser_type::const_iterator;

    bool next()
    {
      auto logFailure = [this](const std::string& msg, const std::runtime_error& e) {
        if (!this->mExtFailureCounter || (*this->mExtFailureCounter)++ < this->mMaxFailureMessages) {
          if (this->mSeverity == fair::Severity::alarm) {
            LOG(alarm) << msg << (*this->mInputIterator).spec->binding << " : " << e.what();
          } else if (this->mSeverity == fair::Severity::warn) {
            LOG(warn) << msg << (*this->mInputIterator).spec->binding << " : " << e.what();
          } else if (this->mSeverity == fair::Severity::fatal) {
            LOG(fatal) << msg << (*this->mInputIterator).spec->binding << " : " << e.what();
          } else if (this->mSeverity == fair::Severity::info) {
            LOG(info) << msg << (*this->mInputIterator).spec->binding << " : " << e.what();
          } else {
            LOG(debug) << msg << (*this->mInputIterator).spec->binding << " : " << e.what();
          }
        }
      };

      while (mInputIterator != mEnd) {
        bool isInitial = mParser == nullptr;
        while (mPartIterator != mInputIterator.end()) {
          // first increment on the parser level
          if (mParser && mCurrent != mParser->end() && ++mCurrent != mParser->end()) {
            // we have an active parser and there is still data at the incremented iterator
            return true;
          }
          // now increment on the level of one input
          mParser.reset();
          if (!isInitial && (mPartIterator == mInputIterator.end() || ++mPartIterator == mInputIterator.end())) {
            // no more parts, go to next input
            break;
          }
          isInitial = false;
          // check filter rules
          if (mFilterSpecs.size() > 0) {
            bool isSelected = false;
            for (auto const& spec : mFilterSpecs) {
              if ((isSelected = DataRefUtils::match(*mPartIterator, spec)) == true) {
                break;
              }
            }
            if (!isSelected) {
              continue;
            }
          }
          gsl::span<const char> raw;
          try {
            raw = mParent.get<gsl::span<char>>(*mPartIterator);
          } catch (const std::runtime_error& e) {
            logFailure("failed to read data from ", e);
          }
          if (raw.size() == 0) {
            continue;
          }

          try {
            mParser = std::make_unique<parser_type>(raw.data(), raw.size());
          } catch (const std::runtime_error& e) {
            logFailure("can not create raw parser from ", e);
          }

          if (mParser != nullptr) {
            mCurrent = mParser->begin();
            return true;
          }
        } // end loop over parts on one input
        ++mInputIterator;
        mPartIterator = mInputIterator.begin();
      } // end loop over inputs
      return false;
    }

    InputRecord& mParent;
    input_iterator mInputIterator;
    input_iterator mEnd;
    part_iterator mPartIterator;
    std::unique_ptr<parser_type> mParser;
    parser_iterator mCurrent;
    std::vector<InputSpec> const& mFilterSpecs;
    size_t mMaxFailureMessages = -1;
    size_t* mExtFailureCounter = nullptr; // external optionally provided counter to throttle error messages
    fair::Severity mSeverity = fair::Severity::alarm;
  };

  using const_iterator = Iterator<DataRef const>;

  const_iterator begin() const
  {
    return const_iterator(mInputs, mInputs.begin(), mInputs.end(), mFilterSpecs, mSeverity, mMaxFailureMessages, mExtFailureCounter);
  }

  const_iterator end() const
  {
    return const_iterator(mInputs, mInputs.end(), mInputs.end(), mFilterSpecs, mSeverity, mMaxFailureMessages, mExtFailureCounter);
  }

  /// Format helper for stream output of the iterator content,
  /// print RDH version and table header
  using RDHInfo = typename o2::framework::DPLRawParser<BOUNDS_CHECKS>::const_iterator::template Fmt<raw_parser::FormatSpec::Info>;

 private:
  InputRecord& mInputs;
  std::vector<InputSpec> mFilterSpecs;
  size_t mMaxFailureMessages = -1;
  size_t* mExtFailureCounter = nullptr; // external optionally provided counter to throttle error messages
  fair::Severity mSeverity = fair::Severity::alarm;
};

} // namespace o2::framework

#endif //FRAMEWORK_UTILS_DPLRAWPARSER_H
