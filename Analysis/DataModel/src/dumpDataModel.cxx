// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/AnalysisDataModel.h"
#include "Analysis/SecondaryVertex.h"
#include <fmt/printf.h>
#include <map>

using namespace o2::framework;
using namespace o2::aod;
using namespace o2::soa;

static int count = 0;

template <typename C>
void printColumn()
{
  if constexpr (!is_index_column_v<C>) {
    fmt::printf("%s\\l", C::label());
  }
}

template <typename C>
void printIndexColumn()
{
  if constexpr (is_index_column_v<C>) {
    fmt::printf("%s\\l", C::label());
  }
}

template <typename C, typename T>
void printIndex()
{
  auto a = MetadataTrait<typename C::binding_t>::metadata::label();
  auto b = MetadataTrait<T>::metadata::label();
  fmt::printf("%s -> %s []\n", a, b);
}

template <typename... C>
void dumpColumns(pack<C...>)
{
  (printColumn<C>(), ...);
  fmt::printf("%s", "\n");
}

template <typename... C>
void dumpIndexColumns(pack<C...>)
{
  (printIndexColumn<C>(), ...);
  fmt::printf("%s", "\n");
}

template <typename T, typename... C>
void dumpIndex(pack<C...>)
{
  (printIndex<C, T>(), ...);
  fmt::printf("%s", "\n");
}

template <typename T>
void dumpTable(bool index = true)
{
  //  nodes.push_back({MetadataTrait<T>::metadata::label(), nodeCount});
  auto label = MetadataTrait<T>::metadata::label();
  fmt::printf(R"(%s[label = "{%s|)", label, label);
  if (pack_size(typename T::iterator::persistent_columns_t{}) -
      pack_size(typename T::iterator::external_index_columns_t{})) {
    dumpColumns(typename T::iterator::persistent_columns_t{});
    fmt::printf("%s", "|");
  }
  if (pack_size(typename T::iterator::dynamic_columns_t{})) {
    dumpColumns(typename T::iterator::dynamic_columns_t{});
    fmt::printf("%s", "|");
  }
  dumpIndexColumns(typename T::iterator::external_index_columns_t{});
  fmt::printf("%s", "}\"]\n");
  if (index)
    dumpIndex<T>(typename T::iterator::external_index_columns_t{});
}

template <typename... Ts>
void dumpCluster()
{
  fmt::printf(R"(subgraph cluster_%d {
node[shape=record,style=filled,fillcolor=gray95]
edge[dir=back, arrowtail=empty]
)",
              count++);
  (dumpTable<Ts>(false), ...);
  fmt::printf("%s", "}\n");
  (dumpIndex<Ts>(typename Ts::iterator::external_index_columns_t{}), ...);
}

int main(int argc, char** argv)
{
  fmt::printf("%s", R"(digraph hierarchy {
size="5,5"
node[shape=record,style=filled,fillcolor=gray95]
edge[dir=back, arrowtail=empty]
)");
  dumpCluster<Tracks, TracksCov, TracksExtra>();
  dumpTable<Collisions>();
  dumpTable<Calos>();
  dumpTable<CaloTriggers>();
  dumpTable<Muons>();
  dumpTable<MuonClusters>();
  dumpTable<Zdcs>();
  dumpTable<VZeros>();
  dumpTable<V0s>();
  dumpTable<Cascades>();
  dumpTable<Timeframes>();
  dumpTable<SecVtx2Prong>();
  dumpTable<Cand2Prong>();
  fmt::printf("%s\n", R"(})");
}
