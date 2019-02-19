#ifndef ROOT_RCOMBINEDDS
#define ROOT_RCOMBINEDDS

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDataSource.hxx"

#include <memory>
#include <string>
#include <vector>

namespace ROOT
{

namespace RDF
{

class RCombinedDSIndex
{
 public:
  virtual ~RCombinedDSIndex() = default;
  /// This is invoked on every Inititialise of the RCombinedDS to
  /// allow constructing the index associated to it.
  /// \param[in]left is the dataframe constructed on top of the left input.
  /// \param[in]right is the dataframe constructed on top of the right input.
  virtual std::vector<std::pair<ULong64_t, ULong64_t>> BuildIndex(std::unique_ptr<RDataFrame>& left,
                                                                  std::unique_ptr<RDataFrame>& right) = 0;
  /// This is invoked on every GetEntry() of the RCombinedDS and
  /// it's used to effectively enumerate all the pairs of the combination.
  virtual std::pair<ULong64_t, ULong64_t> GetAssociatedEntries(ULong64_t entry) = 0;
};

/// An index which allows doing a cross join between two tables.
class RCombindedDSCrossJoinIndex : public RCombinedDSIndex
{
 public:
  std::pair<ULong64_t, ULong64_t> GetAssociatedEntries(ULong64_t entry) final
  {
    return std::make_pair<ULong64_t, ULong64_t>(entry / fRightCount, entry % fRightCount);
  }
  std::vector<std::pair<ULong64_t, ULong64_t>> BuildIndex(std::unique_ptr<RDataFrame>& left,
                                                          std::unique_ptr<RDataFrame>& right) final;

 private:
  ULong64_t fLeftCount;
  ULong64_t fRightCount;
};

/// An index which allows doing a join using a column on the right.
/// FIXME: the need for templation is due to the fact RDataFrame::GetColumnType
///        was introduced only in ROOT 6.16.x. We can remove it once
///        we have a proper build of ROOT.
template <typename INDEX_TYPE = int>
class RCombindedDSColumnJoinIndex : public RCombinedDSIndex
{
 public:
  RCombindedDSColumnJoinIndex(std::string const& indexColumnName)
    : fIndexColumnName{ indexColumnName }
  {
  }

  std::pair<ULong64_t, ULong64_t> GetAssociatedEntries(ULong64_t entry) final
  {
    auto left = fAssociations[entry];
    return std::pair<ULong64_t, ULong64_t>(left, entry);
  }

  std::vector<std::pair<ULong64_t, ULong64_t>>
    BuildIndex(std::unique_ptr<RDataFrame>& left,
               std::unique_ptr<RDataFrame>& right) final
  {
    std::vector<std::pair<ULong64_t, ULong64_t>> ranges;
    auto nEntries = *right->Count();
    fAssociations.reserve(nEntries);
    // Fill the index with the associations
    auto filler = [& assoc = fAssociations](INDEX_TYPE ri) { assoc.push_back(ri); };
    right->Foreach(filler, std::vector<std::string>{ fIndexColumnName });

    // Create the ranges by processing 64 entries per range
    auto deltaRange = 64;
    ranges.reserve(nEntries / deltaRange + 1);
    ULong64_t i = 0;
    while (deltaRange * (i + 1) < nEntries) {
      ranges.emplace_back(std::pair<ULong64_t, ULong64_t>(deltaRange * i, deltaRange * (i + 1)));
    }
    ranges.emplace_back(std::pair<ULong64_t, ULong64_t>(deltaRange * i, nEntries)); // Last entry
    return ranges;
  }

 private:
  std::string fIndexColumnName;
  std::vector<INDEX_TYPE> fAssociations;
};

/// An RDataSource which combines other two
/// RDataSource doing a join operation between them.
class RCombinedDS final : public ROOT::RDF::RDataSource
{
 private:
  /// We need bare pointers because we need to move the ownership of
  /// the datasource to the dataframe
  RDataSource* fLeft;
  RDataSource* fRight;
  std::string fLeftPrefix;
  std::string fRightPrefix;
  std::unique_ptr<RDataFrame> fLeftDF;
  std::unique_ptr<RDataFrame> fRightDF;
  ULong64_t fLeftCount;
  ULong64_t fRightCount;
  size_t fNSlots = 0U;
  std::vector<std::string> fColumnNames;
  std::vector<std::pair<ULong64_t, ULong64_t>> fEntryRanges;
  std::unique_ptr<RCombinedDSIndex> fIndex;

 protected:
  std::vector<void*> GetColumnReadersImpl(std::string_view colName, const std::type_info& info) override;

 public:
  /// Constructs a data source which is given by all the pairs between the elements of left and right
  RCombinedDS(std::unique_ptr<RDataSource> left,
              std::unique_ptr<RDataSource> right,
              std::unique_ptr<RCombinedDSIndex> index = std::make_unique<RCombindedDSCrossJoinIndex>(),
              std::string leftPrefix = std::string{ "left_" },
              std::string rightPrefix = std::string{ "right_" });
  ~RCombinedDS() override;

  template <typename T>
  std::vector<T**> GetColumnReaders(std::string_view colName)
  {
    if (colName.compare(0, fLeftPrefix.size(), fLeftPrefix)) {
      colName.remove_prefix(fLeftPrefix.size());
      return fLeft->GetColumnReaders<T>(colName);
    }
    if (colName.compare(0, fRightPrefix.size(), fRightPrefix)) {
      colName.remove_prefix(fRightPrefix.size());
      return fRight->GetColumnReaders<T>(colName);
    }
    throw std::runtime_error("Column not found: " + colName);
  }
  const std::vector<std::string>& GetColumnNames() const override;
  std::vector<std::pair<ULong64_t, ULong64_t>> GetEntryRanges() override;
  std::string GetTypeName(std::string_view colName) const override;
  bool HasColumn(std::string_view colName) const override;
  bool SetEntry(unsigned int slot, ULong64_t entry) override;
  void InitSlot(unsigned int slot, ULong64_t firstEntry) override;
  void SetNSlots(unsigned int nSlots) override;
  void Initialise() override;
};

////////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Factory method to create a Apache Arrow RDataFrame.
/// \param[in] table an apache::arrow table to use as a source.
RDataFrame MakeCombinedDataFrame(std::unique_ptr<RDataSource> left, std::unique_ptr<RDataSource>, std::unique_ptr<RCombinedDSIndex> index, std::string leftPrefix = "left_", std::string rightPrefix = "right_");
RDataFrame MakeCrossProductDataFrame(std::unique_ptr<RDataSource> left, std::unique_ptr<RDataSource>, std::string leftPrefix = "left_", std::string rightPrefix = "right_");
RDataFrame MakeColumnIndexedDataFrame(std::unique_ptr<RDataSource> left, std::unique_ptr<RDataSource>, std::string indexColName, std::string leftPrefix = "left_", std::string rightPrefix = "right_");

} // namespace RDF

} // namespace ROOT

#endif
