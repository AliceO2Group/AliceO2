// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_INPUTRECORDWALKER_H
#define FRAMEWORK_INPUTRECORDWALKER_H

/// @file   InputRecordWalker.h
/// @author Matthias Richter
/// @since  2020-03-25
/// @brief  A helper class to iteratate over all parts of all input routes

#include "Framework/InputRecord.h"

namespace o2::framework
{

/// @class InputRecordWalker
/// @brief A helper class to iteratate over all parts of all input routes
///
/// Each input route can have multiple parts per input route and computation.
/// This class allows simple iteration over all parts available in the computation.
/// Optionally, a filter can be used to define valid parts and ignore all parts not
/// matching the filter.
///
/// The iterator has DataRef as value type.
///
/// Usage:
///   // 'inputs' refers to the DPL InputRecord instance returned by
///   // ProcessingContext::inputs()
///
///   // iterate over all parts
///   for (auto const& ref : InputRecordWalker(inputs)) {
///     // do something with the data described by the DataRef object, e.g.
///     auto data = inputs.get<TYPE>(ref)
///   }
///
///   // iterate with a filter
///   std::vector<InputSpec> filter{
///     {"tpc", "TPC", "SOMEDATA", 0, Lifetime::Timeframe},
///     {"its", ConcreteDataTypeMatcher{"ITS", "SOMEDATA"}, Lifetime::Timeframe},
///   };
///   for (auto const& ref : InputRecordWalker(inputs, filter)) {
///     // do something with the data
///   }
class InputRecordWalker
{
 public:
  InputRecordWalker() = delete;
  InputRecordWalker(InputRecord& record, std::vector<InputSpec> filterSpecs = {}) : mRecord(record), mFilterSpecs(filterSpecs) {}

  template <typename T>
  using IteratorBase = std::iterator<std::forward_iterator_tag, T>;

  /// Iterator implementation
  /// Supports the following operations:
  /// - increment (there is no decrement, its not a bidirectional parser)
  /// - dereference operator returns @a RawDataHeaderInfo as common header
  template <typename T>
  class Iterator : public IteratorBase<T>
  {
   public:
    using self_type = Iterator;
    using value_type = typename IteratorBase<T>::value_type;
    using reference = typename IteratorBase<T>::reference;
    using pointer = typename IteratorBase<T>::pointer;
    // the iterator over the input routes
    using input_iterator = decltype(std::declval<InputRecord>().begin());

    Iterator() = delete;

    Iterator(InputRecord& parent, input_iterator it, input_iterator end, std::vector<InputSpec> const& filterSpecs)
      : mParent(parent), mInputIterator(it), mEnd(end), mCurrent(mInputIterator.begin()), mFilterSpecs(filterSpecs)
    {
      next(true);
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
      result = result && mCurrent == other.mCurrent;
      return result;
    }

    bool operator!=(const self_type& rh) const
    {
      return not operator==(rh);
    }

   private:
    // the iterator over the parts in one channel
    using part_iterator = typename input_iterator::const_iterator;

    bool next(bool isInitialPart = false)
    {
      while (mInputIterator != mEnd) {
        while (mCurrent != mInputIterator.end()) {
          // increment on the level of one input
          if (!isInitialPart && (mCurrent == mInputIterator.end() || ++mCurrent == mInputIterator.end())) {
            // no more parts, go to next input
            break;
          }
          isInitialPart = false;
          // check filter rules
          if (mFilterSpecs.size() > 0) {
            bool isSelected = false;
            for (auto const& spec : mFilterSpecs) {
              if ((isSelected = DataRefUtils::match(*mCurrent, spec)) == true) {
                break;
              }
            }
            if (!isSelected) {
              continue;
            }
          }
          return true;
        }
        ++mInputIterator;
        mCurrent = mInputIterator.begin();
        isInitialPart = true;
      } // end loop over record
      return false;
    }

    InputRecord& mParent;
    input_iterator mInputIterator;
    input_iterator mEnd;
    part_iterator mCurrent;
    std::vector<InputSpec> const& mFilterSpecs;
  };

  using const_iterator = Iterator<DataRef const>;

  const_iterator begin() const
  {
    return const_iterator(mRecord, mRecord.begin(), mRecord.end(), mFilterSpecs);
  }

  const_iterator end() const
  {
    return const_iterator(mRecord, mRecord.end(), mRecord.end(), mFilterSpecs);
  }

 private:
  InputRecord& mRecord;
  std::vector<InputSpec> mFilterSpecs;
};

} // namespace o2::framework

#endif // FRAMEWORK_INPUTRECORDWALKER_H
