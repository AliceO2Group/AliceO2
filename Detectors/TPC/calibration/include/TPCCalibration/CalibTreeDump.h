// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TPC_CALIBTREEDUMP_H_
#define ALICEO2_TPC_CALIBTREEDUMP_H_

#include <vector>
#include <string>

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
  using DataTypes = boost::variant<CalDet<int>, CalDet<float>, CalDet<double>, CalDet<bool>, CalDet<unsigned int>>;

  CalibTreeDump() = default;
  ~CalibTreeDump() = default;

  /// Add CalDet object
  template <typename T>
  void add(const CalDet<T>& calDet)
  {
    mCalDetObjects.push_back(calDet);
  }
  //void add(const CalDet<DataTypes>& calDet) { mCalDetObjects.push_back(calDet); }

  /// Add CalArray objects
  template <typename T>
  void add(const CalArray<T>& calArray)
  {
    mCalArrayObjects.push_back(calArray);
  }

  /// Dump the registered calibration data to file
  void dumpToFile(const std::string filename = "CalibTree.root");

 private:
  std::vector<DataTypes> mCalDetObjects;   ///< array of CalDet objects
  std::vector<DataTypes> mCalArrayObjects; ///< array of CalArray objects

  /// add default mapping like local, global x/y positions
  void addDefaultMapping(TTree* tree);

  /// add the values of the calDet objects
  /// \todo still to be finalized
  void addCalDetObjects(TTree* tree);

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
