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

 protected:
  std::vector<void*> GetColumnReadersImpl(std::string_view colName, const std::type_info& info) override;

 public:
  /// Constructs a data source which is given by all the pairs between the elements of left and right
  RCombinedDS(std::unique_ptr<RDataSource> left, std::unique_ptr<RDataSource> right, std::string leftPrefix = std::string{ "left_" }, std::string rightPrefix = std::string{ "right_" });
  ~RCombinedDS();

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
RDataFrame MakeCombinedDataFrame(std::shared_ptr<RCombinedDS> left, std::shared_ptr<RCombinedDS>, std::string leftPrefix = "left_", std::string rightPrefix = "right_");

} // namespace RDF

} // namespace ROOT

#endif
