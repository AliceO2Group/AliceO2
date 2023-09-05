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

#ifndef ALICEO2_TPC_CALIBTREEDUMP_H_
#define ALICEO2_TPC_CALIBTREEDUMP_H_

#include <vector>
#include <string>
#include <unordered_map>

#include <boost/variant.hpp>

/// \file   CalibTreeDump.h
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

// forward declarations
class TTree;

namespace o2
{
namespace tpc
{

// forward declarations
template <class T>
class CalDet;

template <class T>
class CalArray;

/// \brief class to dump calibration data to a ttree for simple visualisation
///
/// This class class provided functionanlity for dumping calibration data from
/// CalDet and CalArray objects to a TTree for easy visualisation
///
/// origin: TPC
/// \todo At present this will only work for the PadSubset type ROC
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

class CalibTreeDump
{
 public:
  using DataTypes = CalDet<float>; // boost::variant<CalDet<int>, CalDet<float>, CalDet<double>, CalDet<bool>, CalDet<unsigned int>>;
  using CalPadMapType = std::unordered_map<std::string, DataTypes>;

  CalibTreeDump() = default;
  ~CalibTreeDump() = default;

  /// Add CalDet object
  template <typename T>
  void add(CalDet<T>* calDet)
  {
    mCalDetObjects.emplace_back(calDet);
  }
  // void add(const CalDet<DataTypes>& calDet) { mCalDetObjects.push_back(calDet); }

  /// Add CalArray objects
  template <typename T>
  void add(CalArray<T>* calArray)
  {
    mCalArrayObjects.emplace_back(calArray);
  }
  ///
  /// Add map of CalArray objects changes the name of the CalDet to the map name
  /// to have unique identifier
  void add(CalPadMapType& calibs);

  /// Set adding of FEE mapping to the tree
  void setAddFEEInfo(bool add = true) { mAddFEEInfo = add; }

  /// Add CalPad objects from a file
  void addCalPads(const std::string_view file, const std::string_view calPadNames);

  /// Dump the registered calibration data to file
  void dumpToFile(const std::string filename = "CalibTree.root");

  /// Add complementary information
  void addInfo(const std::string_view name, float value) { mAddInfo[name.data()] = value; }

 private:
  std::unordered_map<std::string, float> mAddInfo{}; ///< additional common information to be added to the output tree
  std::vector<DataTypes*> mCalDetObjects{};          ///< array of CalDet objects
  std::vector<DataTypes*> mCalArrayObjects{};        ///< array of CalArray objects
  bool mAddFEEInfo{false};                           ///< add front end electronics mappings
  std::vector<float> mTraceLengthIROC;               ///< trace lengths IROC
  std::vector<float> mTraceLengthOROC;               ///< trace lengths OROC

  /// add default mapping like local, global x/y positions
  void addDefaultMapping(TTree* tree);

  /// add FEE mapping like FEC id, SAMPA id, chip id
  void addFEEMapping(TTree* tree);

  /// add the values of the calDet objects
  /// \todo still to be finalized
  void addCalDetObjects(TTree* tree);

  /// set default aliases
  void setDefaultAliases(TTree* tree);

  /// forwarding visitor class, required to loop over the variant types
  template <class Result, class Func>
  struct forwarding_visitor : boost::static_visitor<Result> {
    Func func;
    forwarding_visitor(const Func& f) : func(f) {}
    forwarding_visitor(Func&& f) : func(std::move(f)) {}
    template <class Arg>
    Result operator()(Arg&& arg) const
    {
      return func(std::forward<Arg>(arg));
    }
  };

  /// easy way to make a forwarding visitor
  template <class Result, class Func>
  forwarding_visitor<Result, std::decay_t<Func>> make_forwarding_visitor(Func&& func)
  {
    return {std::forward<Func>(func)};
  }
};
} // namespace tpc

} // namespace o2
#endif
