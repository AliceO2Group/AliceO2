// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALGORITHM_TABLEVIEW_H
#define ALGORITHM_TABLEVIEW_H

/// @file   TableView.h
/// @author Matthias Richter
/// @since  2017-09-21
/// @brief  Container class for multiple sequences of data wrapped by markers

#include <vector>
#include <map>

namespace o2
{

namespace algorithm
{

/**
 * @class TableView
 * Container class for multiple sequences of data wrapped by markers.
 *
 * This is a container for data sequences of multiple frames consisting
 * of a header marker struct, a payload, and an optional trailer marker
 * struct. Each sequence forms a row in the TableView, the columns
 * are provided by the markers/frames.
 *
 * A parser is used to step through the data sequence and extract
 * headers, trailers and payload positions.
 *
 * Requirements:
 * - both header and trailer type must provide an operator bool() method
 *   to check validity
 * - the size of one frame needs to be extracted either from the header
 *   marker or the trailer marker. The first requires forward, the latter
 *   backward parsing. In the first case, the trailer is optional, while
 *   in the latter required
 *
 */
template <typename RowDescT,    // row description
          typename ColumnDescT, // column description
          typename ParserT      // parser type (forward/backward)
          >
class TableView
{
 public:
  TableView() = default;
  ~TableView() = default;

  using RowDescType = RowDescT;
  using ColumnIndexType = ColumnDescT;
  using ParserType = ParserT;

  /// FrameIndex is composed from column description and row number
  struct FrameIndex {
    ColumnIndexType columnIndex;
    unsigned row;

    bool operator<(const FrameIndex& rh) const
    {
      if (rh.columnIndex < columnIndex)
        return false;
      if (columnIndex < rh.columnIndex)
        return true;
      return row < rh.row;
    }
  };

  /// descriptor pointing to payload of one frame
  struct FrameData {
    const byte* buffer = nullptr;
    size_t size = 0;
  };

  /**
   * Add a new data sequence, the set is traversed according to parser
   *
   * TODO: functors to check header and trailer validity as well as retrieving
   * the frame size could be passed as arguments.
   *
   * @param rowData   Descriptive data struct for the sequence
   * @param seqData    Pointer to sequence
   * @param seqSize    Length of sequence
   * @return number of inserted elements
   */
  size_t addRow(RowDescType rowData, byte* seqData, size_t seqSize)
  {
    unsigned nFrames = mFrames.size();
    unsigned currentRow = mRowData.size();
    ParserType p;
    p.parse(
      seqData, seqSize,
      [](const typename ParserT::HeaderType& h) { return (h); },
      [](const typename ParserT::TrailerType& t) { return (t); },
      [](const typename ParserT::TrailerType& t) {
        return t.dataLength + ParserT::totalOffset;
      },
      [this, currentRow](typename ParserT::FrameInfo entry) {
        // insert the header as column index in ascending order
        auto position = mColumns.begin();
        while (position != mColumns.end() && *position < *entry.header) {
          position++;
        }
        if (position == mColumns.end() || *entry.header < *position) {
          mColumns.emplace(position, *entry.header);
        }

        // insert frame descriptor under key composed from header and row
        auto result = mFrames.emplace(FrameIndex{*entry.header, currentRow},
                                      FrameData{entry.payload, entry.length});
        return result.second;
      });
    auto insertedFrames = mFrames.size() - nFrames;
    if (insertedFrames > 0) {
      mRowData.emplace_back(rowData);
    }
    return insertedFrames;
  }

  /// clear the index, i.e. all internal lists
  void clear()
  {
    mFrames.clear();
    mColumns.clear();
    mRowData.clear();
  }

  /// get number of columns in the created index
  size_t getNColumns() const { return mColumns.size(); }

  /// get number of rows, i.e. number rows in the created index
  size_t getNRows() const { return mRowData.size(); }

  /// get row data for a data set
  const RowDescType& getRowData(size_t row) const
  {
    if (row < mRowData.size())
      return mRowData[row];
    // TODO: better to throw exception?
    static RowDescType dummy;
    return dummy;
  }

  // TODO:
  // instead of a member with this pointer of parent class, the access
  // function was supposed to be specified as a lambda. This definition
  // was supposed to be the type of the function member.
  // passing the access function to the iterator did not work because
  // the typedef for the access function is without the capture, so there
  // is no matching conversion.
  // Solution would be to use std::function but that's probably slow and
  // the function is called often. Can be checked later.
  typedef FrameData (*AccessFct)(unsigned, unsigned);

  /// Iterator class for configurable direction, i.e. either row or column
  class iterator
  { // TODO: derive from forward_iterator
   public:
    struct value_type : public FrameData {
      RowDescType desc;
    };
    using self_type = iterator;

    enum IteratorDirections {
      kAlongRow,
      kAlongColumn
    };

    iterator() = delete;
    ~iterator() = default;
    iterator(IteratorDirections direction, TableView* parent, unsigned row = 0, unsigned column = 0)
      : mDirection(direction), mRow(row), mColumn(column), mEnd(direction == kAlongRow ? parent->getNColumns() : parent->getNRows()), mParent(parent), mCache(), mIsCached(false)
    {
      while (!isValid() && !isEnd())
        operator++();
    }

    self_type& operator++()
    {
      mIsCached = false;
      if (mDirection == kAlongRow) {
        if (mColumn < mEnd)
          mColumn++;
      } else {
        if (mRow < mEnd)
          mRow++;
      }
      while (!isEnd() && !isValid())
        operator++();
      return *this;
    }

    value_type operator*() const
    {
      if (!mIsCached) {
        self_type* ncthis = const_cast<self_type*>(this);
        mParent->get(mRow, mColumn, ncthis->mCache);
        ncthis->mCache.desc = mParent->getRowData(mRow);
        ncthis->mIsCached = true;
      }
      return mCache;
    }

    bool operator==(const self_type& other) const
    {
      return mDirection == kAlongRow ? (mColumn == other.mColumn) : (mRow == other.mRow);
    }

    bool operator!=(const self_type& other) const
    {
      return mDirection == kAlongRow ? (mColumn != other.mColumn) : (mRow != other.mRow);
    }

    bool isEnd() const
    {
      return (mDirection == kAlongRow) ? (mColumn >= mEnd) : (mRow >= mEnd);
    }

    bool isValid() const
    {
      if (!mIsCached) {
        self_type* ncthis = const_cast<self_type*>(this);
        ncthis->mIsCached = mParent->get(mRow, mColumn, ncthis->mCache);
        ncthis->mCache.desc = mParent->getRowData(mRow);
      }
      return mIsCached;
    }

   protected:
    IteratorDirections mDirection;
    unsigned mRow;
    unsigned mColumn;
    unsigned mEnd;
    TableView* mParent;
    value_type mCache;
    bool mIsCached;
  };

  /// iterator for the outer access of the index, either row or column direction
  template <unsigned Direction>
  class outerIterator : public iterator
  {
   public:
    using base = iterator;
    using value_type = typename base::value_type;
    using self_type = outerIterator;
    static const unsigned direction = Direction;

    outerIterator() = delete;
    ~outerIterator() = default;
    outerIterator(TableView* parent, unsigned index)
      : iterator(typename iterator::IteratorDirections(direction), parent, direction == iterator::kAlongColumn ? index : 0, direction == iterator::kAlongRow ? index : 0)
    {
    }

    self_type& operator++()
    {
      if (base::mDirection == iterator::kAlongRow) {
        if (base::mColumn < base::mEnd)
          base::mColumn++;
      } else {
        if (base::mRow < base::mEnd)
          base::mRow++;
      }
      return *this;
    }

    /// begin the inner iteration
    iterator begin()
    {
      return iterator((base::mDirection == iterator::kAlongColumn) ? iterator::kAlongRow : iterator::kAlongColumn,
                      base::mParent,
                      (base::mDirection == iterator::kAlongColumn) ? base::mRow : 0,
                      (base::mDirection == iterator::kAlongRow) ? base::mColumn : 0);
    }

    /// end of the inner iteration
    iterator end()
    {
      return iterator((base::mDirection == iterator::kAlongColumn) ? iterator::kAlongRow : iterator::kAlongColumn,
                      base::mParent,
                      (base::mDirection == iterator::kAlongRow) ? base::mParent->getNRows() : 0,
                      (base::mDirection == iterator::kAlongColumn) ? base::mParent->getNColumns() : 0);
    }
  };

  /// definition of the outer iterator over column
  using ColumnIterator = outerIterator<iterator::kAlongRow>;
  /// definition of the outer iterator over row
  using RowIterator = outerIterator<iterator::kAlongColumn>;

  /// begin of the outer iteration
  ColumnIterator begin()
  {
    return ColumnIterator(this, 0);
  }

  /// end of outer iteration
  ColumnIterator end()
  {
    return ColumnIterator(this, mColumns.size());
  }

 private:
  /// private access function for the iterators
  bool get(unsigned row, unsigned column, FrameData& data)
  {
    if (this->mColumns.size() == 0)
      return false;
    auto element = this->mFrames.find(FrameIndex{this->mColumns[column], row});
    if (element != this->mFrames.end()) {
      data = element->second;
      return true;
    }
    return false;
  }

  /// map of frame descriptors with key composed from header and row number
  std::map<FrameIndex, FrameData> mFrames;
  /// list of indices in row direction
  std::vector<ColumnIndexType> mColumns;
  /// data descriptor of each row forming the columns
  std::vector<RowDescType> mRowData;
};

} // namespace algorithm

} // namespace o2

#endif // ALGORITHM_TABLEVIEW_H
