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
#include "Analysis/Jet.h"
#include <fmt/printf.h>
#include <map>

using namespace o2::framework;
using namespace o2::aod;
using namespace o2::soa;

static int count = 0;

template <typename C>
void printColumn(char const* fg, char const* bg)
{
  if constexpr (!is_index_column_v<C>) {
    fmt::printf("<TR><TD color='%s' bgcolor='%s'>%s</TD></TR>", fg, bg, C::columnLabel());
  }
}

template <typename C>
void printIndexColumn(char const* fg, char const* bg)
{
  if constexpr (is_index_column_v<C>) {
    fmt::printf("<TR><TD color='%s' bgcolor='%s'>%s</TD></TR>", fg, bg, C::columnLabel());
  }
}

template <typename C, typename T>
void printIndex()
{
  if constexpr (!is_type_with_originals_v<typename C::binding_t>) {
    auto a = MetadataTrait<typename C::binding_t>::metadata::tableLabel();
    auto b = MetadataTrait<T>::metadata::tableLabel();
    fmt::printf("%s -> %s []\n", a, b);
  } else {
    using main_original = pack_element_t<0, typename C::binding_t::originals>;
    auto a = MetadataTrait<main_original>::metadata::tableLabel();
    auto b = MetadataTrait<T>::metadata::tableLabel();
    fmt::printf("%s -> %s []\n", a, b);
  }
}

template <typename... C>
void dumpColumns(pack<C...>, const char* fg, const char* bg)
{
  (printColumn<C>(fg, bg), ...);
  fmt::printf("%s", "\n");
}

template <typename... C>
void dumpIndexColumns(pack<C...>, char const* fg, char const* bg)
{
  (printIndexColumn<C>(fg, bg), ...);
  fmt::printf("%s", "\n");
}

template <typename T, typename... C>
void dumpIndex(pack<C...>)
{
  (printIndex<C, T>(), ...);
  fmt::printf("%s", "\n");
}

struct Style {
  const char* color;
  const char* background;
  const char* fontcolor;
  const char* headerfontcolor;
  const char* headerbgcolor;
  const char* methodcolor;
  const char* methodbgcolor;
  const char* indexcolor;
  const char* indexbgcolor;
};

static Style styles[] = {
  {"black", "gray80", "black", "black", "gray70", "black", "gray60", "black", "gray50"},
  {"/reds9/2", "/reds9/4", "white", "white", "/reds9/7", "black", "/reds9/6", "/reds9/1", "/reds9/5"},
  {"/greens9/2", "/greens9/4", "white", "white", "/greens9/7", "black", "/greens9/6", "/greens9/1", "/greens9/5"},
  {"/blues9/2", "/blues9/4", "white", "white", "/blues9/7", "black", "/blues9/6", "/blues9/1", "/blues9/5"},
};

Style const& getDefaultStyle()
{
  return styles[0];
}

enum struct StyleType : int {
  DEFAULT = 0,
  RED = 1,
  GREEN = 2,
  BLUE = 3,
};

template <typename T>
void dumpTable(bool index = true, enum StyleType styleId = StyleType::DEFAULT)
{
  auto style = styles[static_cast<int>(styleId)];
  //  nodes.push_back({MetadataTrait<T>::metadata::label(), nodeCount});
  auto label = MetadataTrait<T>::metadata::tableLabel();
  fmt::printf(R"(%s[color="%s" cellpadding="0" fillcolor="%s" fontcolor="%s" label = <
<TABLE cellpadding='2' cellspacing='0' cellborder='0' ><TH cellpadding='0' bgcolor="black"><TD bgcolor="%s"><font color="%s">%s</font></TD></TH>)",
              label, style.color, style.background, style.fontcolor, style.headerbgcolor, style.headerfontcolor, label);
  if (pack_size(typename T::iterator::persistent_columns_t{}) -
      pack_size(typename T::iterator::external_index_columns_t{})) {
    dumpColumns(typename T::iterator::persistent_columns_t{}, style.color, style.background);
    fmt::printf("%s", "HR");
  }
  if (pack_size(typename T::iterator::dynamic_columns_t{})) {
    dumpColumns(typename T::iterator::dynamic_columns_t{}, style.methodcolor, style.methodbgcolor);
    fmt::printf("%s", "HR");
  }
  dumpIndexColumns(typename T::iterator::external_index_columns_t{}, style.indexcolor, style.indexbgcolor);
  fmt::printf("%s", "</TABLE>\n>]\n");
  if (index)
    dumpIndex<T>(typename T::iterator::external_index_columns_t{});
}

template <typename... Ts>
void dumpCluster()
{
  fmt::printf(R"(subgraph cluster_%d {
node[shape=plain,style=filled,fillcolor=gray95]
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
node[shape=plain,style=filled,fillcolor=gray95]
edge[dir=back, arrowtail=empty]
)");
  dumpCluster<Tracks, TracksCov, TracksExtra>();
  dumpTable<Collisions>();
  dumpTable<Calos>();
  dumpTable<CaloTriggers>();
  dumpTable<Muons>();
  dumpTable<MuonClusters>();
  dumpTable<Zdcs>();
  dumpTable<Run2V0s>();
  dumpTable<V0s>();
  dumpTable<Cascades>();
  dumpTable<BCs>();
  dumpTable<FT0s>();
  dumpTable<FV0s>();
  dumpTable<FDDs>();
  dumpTable<SecVtx2Prong>(true, StyleType::GREEN);
  dumpTable<Cand2Prong>(true, StyleType::GREEN);
  dumpTable<Jets>(true, StyleType::BLUE);
  dumpTable<JetConstituents>(true, StyleType::BLUE);
  dumpTable<UnassignedTracks>();
  dumpTable<McCollisions>(true, StyleType::RED);
  dumpTable<McTrackLabels>(true, StyleType::RED);
  dumpTable<McCaloLabels>(true, StyleType::RED);
  dumpTable<McCollisionLabels>(true, StyleType::RED);
  dumpTable<McParticles>(true, StyleType::RED);
  fmt::printf("%s\n", R"(})");
}
