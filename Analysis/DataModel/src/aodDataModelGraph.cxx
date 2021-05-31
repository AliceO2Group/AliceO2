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
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/PID/PIDResponse.h"
#include "AnalysisDataModel/Multiplicity.h"
#include "AnalysisDataModel/Centrality.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisDataModel/Jet.h"
#include "AnalysisDataModel/StrangenessTables.h"
#include <fmt/printf.h>
#include <map>

using namespace o2::framework;
using namespace o2::aod;
using namespace o2::soa;

static int count = 0;
static int width = 10;
static int height = 10;

void inline graphSize()
{
  fmt::printf(
    R"(
size="%d,%d";
)",
    width, height);
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

enum StyleType : int {
  DEFAULT = 0,
  RED = 1,
  GREEN = 2,
  BLUE = 3,
};

static std::vector<std::pair<std::string, StyleType>> tableStyles = {
  {"HfTrackIndexProng", StyleType::BLUE},
  {"HfCandProng", StyleType::BLUE},
  {"pidResp", StyleType::GREEN},
  {"Mults", StyleType::GREEN},
  {"Cents", StyleType::GREEN},
  {"Timestamps", StyleType::GREEN},
  {"Jet", StyleType::BLUE},
  {"Mc", StyleType::RED},
  {"V0Datas", StyleType::GREEN},
  {"CascData", StyleType::GREEN},
  {"TrackSelection", StyleType::GREEN},
  {"TracksExtended", StyleType::GREEN},
  {"Transient", StyleType::GREEN},
  {"Extension", StyleType::GREEN},
};

template <typename T>
Style getStyleFor()
{
  auto label = MetadataTrait<T>::metadata::tableLabel();
  auto entry = std::find_if(tableStyles.begin(), tableStyles.end(), [&](auto&& x) { if (std::string(label).find(x.first) != std::string::npos) { return true;
}return false; });
  if (entry != tableStyles.end()) {
    auto value = *entry;
    return styles[value.second];
  }
  return styles[StyleType::DEFAULT];
}

void inline nodeEmpty()
{
  fmt::printf(
    R"(node[shape=none,height=0,width=0,label=""])");
}

void inline nodeNormal()
{
  fmt::printf(
    R"(node[shape=plain,style=filled,fillcolor=gray95])");
}

void inline graphHeader(char const* type, char const* name)
{
  fmt::printf(R"(%s %s {
edge[dir=back, arrowtail=empty]
)",
              type, name);
  nodeNormal();
}

void inline graphFooter()
{
  fmt::printf("}\n");
}

template <typename T>
void displayEntity();

template <typename... Ts>
void displayOriginals(pack<Ts...>)
{
  graphHeader("subgraph", fmt::format("cluster_{}", count++).c_str());
  fmt::printf("label = %s;\n", MetadataTrait<pack_element_t<1, pack<Ts...>>>::metadata::tableLabel());
  (..., displayEntity<Ts>());
  graphFooter();
}

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

template <typename... C>
void displayColumns(pack<C...>, const char* fg, const char* bg)
{
  (printColumn<C>(fg, bg), ...);
  fmt::printf("%s", "\n");
}

template <typename... C>
void displayIndexColumns(pack<C...>, char const* fg, char const* bg)
{
  (printIndexColumn<C>(fg, bg), ...);
  fmt::printf("%s", "\n");
}

template <typename C, typename T>
void printIndex()
{
  if constexpr (!is_type_with_originals_v<typename C::binding_t>) {
    auto a = MetadataTrait<typename C::binding_t>::metadata::tableLabel();
    auto b = MetadataTrait<T>::metadata::tableLabel();
    fmt::printf("%s -> %s []\n", a, b);
  } else {
    using main_original = pack_element_t<1, typename C::binding_t::originals>;
    auto a = MetadataTrait<main_original>::metadata::tableLabel();
    auto b = MetadataTrait<T>::metadata::tableLabel();
    fmt::printf("%s -> %s []\n", a, b);
  }
}

template <typename T, typename... C>
void dumpIndex(pack<C...>)
{
  (printIndex<C, T>(), ...);
  fmt::printf("%s", "\n");
}

template <typename T>
void displayTable()
{
  auto style = getStyleFor<T>();
  auto label = MetadataTrait<T>::metadata::tableLabel();
  fmt::printf(R"(%s[color="%s" cellpadding="0" fillcolor="%s" fontcolor="%s" label = <
<TABLE cellpadding='2' cellspacing='0' cellborder='0' ><TH cellpadding='0' bgcolor="black"><TD bgcolor="%s"><font color="%s">%s</font></TD></TH>)",
              label, style.color, style.background, style.fontcolor, style.headerbgcolor, style.headerfontcolor, label);
  if (pack_size(typename T::iterator::persistent_columns_t{}) -
        pack_size(typename T::iterator::external_index_columns_t{}) >
      0) {
    displayColumns(typename T::iterator::persistent_columns_t{}, style.color, style.background);
    fmt::printf("%s", "HR");
  }
  if (pack_size(typename T::iterator::dynamic_columns_t{})) {
    displayColumns(typename T::iterator::dynamic_columns_t{}, style.methodcolor, style.methodbgcolor);
    fmt::printf("%s", "HR");
  }
  displayIndexColumns(typename T::iterator::external_index_columns_t{}, style.indexcolor, style.indexbgcolor);
  fmt::printf("%s", "</TABLE>\n>]\n");
  dumpIndex<T>(typename T::iterator::external_index_columns_t{});
}

template <typename T>
void displayEntity()
{
  if constexpr (is_soa_join_t<T>::value) {
    displayOriginals(typename T::originals{});
  } else {
    displayTable<T>();
  }
}

template <typename... T>
void displayEntities()
{
  graphHeader("subgraph", fmt::format("cluster_{}", count++).c_str());
  (..., displayEntity<T>());
  graphFooter();
}

int main(int, char**)
{
  graphHeader("digraph", "hierarchy");
  graphSize();
  fmt::printf(R"(compound = true;
)");

  displayEntity<BCs>();
  /// rank trick to avoid BCs moving
  nodeEmpty();
  fmt::printf(R"({rank = same; BCs -> root[style=invis];};)");
  nodeNormal();

  displayEntity<Zdcs>();
  displayEntity<FT0s>();
  displayEntity<FV0As>();
  displayEntity<FDDs>();
  displayEntity<HMPIDs>();

  displayEntities<Collisions, Cents, Mults, Timestamps>();
  displayEntity<McCollisions>();
  displayEntity<McCollisionLabels>();

  displayEntity<Calos>();
  displayEntity<CaloTriggers>();
  displayEntity<McCaloLabels>();

  displayEntity<FV0Cs>();
  displayEntity<Run2BCInfos>();

  displayEntities<Tracks, TracksCov, TracksExtra, TracksExtended, TrackSelection,
                  pidTOFFullEl, pidTOFFullMu, pidTOFFullPi,
                  pidTOFFullKa, pidTOFFullPr, pidTOFFullDe,
                  pidTOFFullTr, pidTOFFullHe, pidTOFFullAl,
                  pidTPCFullEl, pidTPCFullMu, pidTPCFullPi,
                  pidTPCFullKa, pidTPCFullPr, pidTPCFullDe,
                  pidTPCFullTr, pidTPCFullHe, pidTPCFullAl>();
  displayEntity<AmbiguousTracks>();
  displayEntity<AmbiguousMFTTracks>();

  displayEntity<McParticles>();
  displayEntity<McTrackLabels>();

  displayEntity<Jets>();
  displayEntity<JetConstituents>();

  displayEntities<V0s, V0Datas>();

  displayEntities<Cascades, CascDataFull>();

  displayEntities<MFTTracks, FwdTracks, FwdTracksCov>();

  displayEntities<HfTrackIndexProng2, HfCandProng2>();
  displayEntities<HfTrackIndexProng3, HfCandProng3>();

  graphFooter();
  return 0;
}
