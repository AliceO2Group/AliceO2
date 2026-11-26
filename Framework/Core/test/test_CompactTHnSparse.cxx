// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/CompactTHnSparse.h"
#include "Framework/HistogramRegistry.h"

#include <TBufferFile.h>
#include <TClass.h>
#include <TH1D.h>
#include <THnSparse.h>
#include <TList.h>
#include <TRandom3.h>

#include <catch_amalgamated.hpp>

#include <limits>
#include <memory>
#include <vector>

namespace
{
constexpr int kDim = 5;
// a variable-width axis, a fine axis that makes dense regions, out-of-range fills (under/overflow)
const Int_t kBins[kDim] = {8, 20, 16, 9, 110};
const Double_t kMin[kDim] = {0., -2., -1.6, 0., -1.05};
const Double_t kMax[kDim] = {8., 2., 4.8, 9., 1.05};
const Double_t kEdges0[9] = {0., 0.2, 0.5, 1., 2., 3., 5., 10., 100.};

template <class H>
void setAxes(H& h)
{
  h.SetBinEdges(0, kEdges0);
  h.GetAxis(1)->SetTitle("#Delta#eta");
}

// Correlated values, so some cells are dense and most of the space stays empty
void draw(TRandom3& r, Double_t* x)
{
  Double_t c = r.Gaus(0., 1.);
  x[0] = std::abs(r.Exp(1.5));
  x[1] = 0.8 * c + r.Gaus(0., 0.3);
  x[2] = r.Uniform(-1.6, 4.8);
  x[3] = r.Uniform(-0.5, 9.5); // under- and overflow
  x[4] = 0.6 * c + r.Gaus(0., 0.4);
}

// Every filled bin of `ref` has the same content and error in `h`, and the statistics agree.
// Errors are compared to 1e-14: where the compiler fuses w * w + sumw2 into one FMA in one of
// the two fill paths but not the other (GCC on aarch64), they differ in the last bit.
void compare(const THnSparse& ref, const THnBase& h)
{
  REQUIRE(h.GetNbins() == ref.GetNbins());
  REQUIRE(h.GetEntries() == ref.GetEntries());
  REQUIRE(h.GetCalculateErrors() == ref.GetCalculateErrors());
  REQUIRE(h.GetSumw() == Catch::Approx(ref.GetSumw()).epsilon(1e-12));
  REQUIRE(h.GetSumw2() == Catch::Approx(ref.GetSumw2()).epsilon(1e-12));
  for (int d = 0; d < kDim; ++d) {
    REQUIRE(h.GetSumwx(d) == Catch::Approx(ref.GetSumwx(d)).epsilon(1e-12));
    REQUIRE(h.GetSumwx2(d) == Catch::Approx(ref.GetSumwx2(d)).epsilon(1e-12));
    REQUIRE(h.GetAxis(d)->GetNbins() == ref.GetAxis(d)->GetNbins());
    REQUIRE(h.GetAxis(d)->GetXmin() == ref.GetAxis(d)->GetXmin());
    REQUIRE(h.GetAxis(d)->GetXmax() == ref.GetAxis(d)->GetXmax());
  }
  Int_t coord[kDim];
  long bad = 0;
  for (Long64_t i = 0; i < ref.GetNbins(); ++i) {
    Double_t v = ref.GetBinContent(i, coord);
    Long64_t b = h.GetBin(coord);
    bad += h.GetBinContent(b) != v || h.GetBinError2(b) != Catch::Approx(ref.GetBinError2(i)).epsilon(1e-14);
  }
  REQUIRE(bad == 0);
}

struct Pair {
  THnSparseF ref{"ref", "ref", kDim, kBins, kMin, kMax};
  CompactTHnSparseF h{"h", "h", kDim, kBins, kMin, kMax};
  Pair()
  {
    setAxes(ref);
    setAxes(h);
  }
  void fill(TRandom3& r, int n, bool weighted)
  {
    Double_t x[kDim];
    for (int i = 0; i < n; ++i) {
      draw(r, x);
      Double_t w = weighted ? r.Uniform(0.1, 3.) : 1.;
      ref.Fill(x, w);
      h.Fill(x, w);
    }
  }
};
} // namespace

TEST_CASE("CompactTHnSparseFillLikeTHnSparse")
{
  TRandom3 r(1);
  Pair p;
  p.fill(r, 200000, false);
  compare(p.ref, p.h);
  REQUIRE(p.h.storage().denseBlocks() > 0); // both block layouts are exercised

  // Errors switched on part way: earlier bins get sumw2 = content, as in THnSparse
  p.ref.Sumw2();
  p.h.Sumw2();
  p.fill(r, 100000, true);
  compare(p.ref, p.h);

  // Variadic fill from THnBase
  p.ref.Fill(0.3, 0.1, 0.2, 4.5, 0.01, 2.5);
  p.h.Fill(0.3, 0.1, 0.2, 4.5, 0.01, 2.5);
  compare(p.ref, p.h);

  // Far less memory than THnSparse needs for its hash table alone
  REQUIRE(p.h.storageBytes() < 30 * size_t(p.h.GetNbins()));
}

TEST_CASE("CompactTHnSparseToTHnSparse")
{
  TRandom3 r(2);
  Pair p;
  p.ref.Sumw2();
  p.h.Sumw2();
  p.fill(r, 100000, true);
  std::unique_ptr<THnSparse> s(p.h.toTHnSparse("converted"));
  REQUIRE(s->IsA() == THnSparseF::Class());
  REQUIRE(std::string(s->GetName()) == "converted");
  REQUIRE(s->GetAxis(0)->GetXbins()->GetSize() == 9);
  REQUIRE(std::string(s->GetAxis(1)->GetTitle()) == "#Delta#eta");
  compare(p.ref, *s);
}

TEST_CASE("CompactTHnSparseMerge")
{
  TRandom3 r(3);
  Pair all;
  CompactTHnSparseF h1("h1", "h1", kDim, kBins, kMin, kMax);
  CompactTHnSparseF h2("h2", "h2", kDim, kBins, kMin, kMax);
  setAxes(h1);
  setAxes(h2);
  Double_t x[kDim];
  for (int i = 0; i < 100000; ++i) {
    draw(r, x);
    all.ref.Fill(x);
    (i % 2 ? h1 : h2).Fill(x);
  }
  TList l;
  l.Add(&h2);
  h1.Merge(&l);
  l.Clear("nodelete");
  compare(all.ref, h1);
}

TEST_CASE("CompactTHnSparseStreamer")
{
  TRandom3 r(4);
  Pair p;
  p.ref.Sumw2();
  p.h.Sumw2();
  p.fill(r, 50000, true);

  TBufferFile buf(TBuffer::kWrite);
  buf.WriteObjectAny(&p.h, CompactTHnSparseF::Class());
  buf.SetReadMode();
  buf.SetBufferOffset(0);
  std::unique_ptr<CompactTHnSparseF> back(static_cast<CompactTHnSparseF*>(buf.ReadObjectAny(CompactTHnSparseF::Class())));
  REQUIRE(back);
  compare(p.ref, *back);

  // Clone goes through the streamer too
  std::unique_ptr<CompactTHnSparseF> clone(static_cast<CompactTHnSparseF*>(p.h.Clone("clone")));
  compare(p.ref, *clone);
}

TEST_CASE("CompactTHnSparseProjectionAndRange")
{
  TRandom3 r(5);
  Pair p;
  p.fill(r, 50000, false);
  std::unique_ptr<TH1D> pr(p.ref.Projection(4));
  std::unique_ptr<TH1D> ph(p.h.Projection(4));
  REQUIRE(ph->GetNbinsX() == pr->GetNbinsX());
  for (int b = 0; b <= pr->GetNbinsX() + 1; ++b) {
    REQUIRE(ph->GetBinContent(b) == pr->GetBinContent(b));
  }
  // Axis ranges are respected by projections (through the bin iterator)
  p.ref.GetAxis(3)->SetRange(2, 5);
  p.h.GetAxis(3)->SetRange(2, 5);
  pr.reset(p.ref.Projection(1));
  ph.reset(p.h.Projection(1));
  REQUIRE(ph->GetEntries() == pr->GetEntries());
  for (int b = 0; b <= pr->GetNbinsX() + 1; ++b) {
    REQUIRE(ph->GetBinContent(b) == pr->GetBinContent(b));
  }
}

TEST_CASE("CompactTHnSparseReset")
{
  TRandom3 r(6);
  Pair p;
  p.fill(r, 1000, true);
  p.h.Reset();
  REQUIRE(p.h.GetNbins() == 0);
  REQUIRE(p.h.GetEntries() == 0);
  Int_t c[kDim] = {1, 1, 1, 1, 1};
  REQUIRE(p.h.GetBinContent(p.h.GetBin(c)) == 0);
}

TEST_CASE("CompactTHnSparseInHistogramRegistry")
{
  using namespace o2::framework;
  std::vector<double> edges0(std::begin(kEdges0), std::end(kEdges0));
  std::vector<AxisSpec> axes{{edges0, "p_{T}"}, {20, -2., 2., "#Delta#eta"}, {16, -1.6, 4.8}, {9, 0., 9.}, {110, -1.05, 1.05}};
  HistogramRegistry registry{"registry"};
  registry.add("ref", "ref", kTHnSparseF, axes, true);
  registry.add("compact/h", "compact", kCompactTHnSparseF, axes, true);
  auto ref = registry.get<THnSparse>(HIST("ref"));
  auto compact = registry.get<CompactTHnSparse>(HIST("compact/h"));
  REQUIRE(compact->IsA() == CompactTHnSparseF::Class());
  REQUIRE(std::string(compact->GetAxis(0)->GetTitle()) == "p_{T}");
  REQUIRE_THROWS(registry.get<THnSparse>(HIST("compact/h")));

  TRandom3 r(7);
  Double_t x[kDim];
  for (int i = 0; i < 50000; ++i) {
    draw(r, x);
    Double_t w = r.Uniform(0.1, 3.);
    registry.fill(HIST("ref"), x[0], x[1], x[2], x[3], x[4], w);
    registry.fill(HIST("compact/h"), x[0], x[1], x[2], x[3], x[4], w);
  }
  compare(*ref, *compact);
  REQUIRE(registry.getSize(HIST("compact/h"), 0.01) < registry.getSize(HIST("ref"), 0.01));

  registry.addClone("compact/", "copy/");
  compare(*ref, *registry.get<CompactTHnSparse>(HIST("copy/h")));

  // What postRunOutput sends for a compact histogram: the equivalent THnSparse, same name
  int converted = 0;
  registry.apply([&](HistogramRegistry const&, TNamed* obj) {
    if (auto* c = dynamic_cast<CompactTHnSparse*>(obj)) {
      std::unique_ptr<THnSparse> s{c->toTHnSparse()};
      REQUIRE(s->IsA() == THnSparseF::Class());
      REQUIRE(std::string(s->GetName()) == c->GetName());
      compare(*ref, *s);
      ++converted;
    }
  });
  REQUIRE(converted == 2);
}

TEST_CASE("CompactTHnSparseOutOfRange")
{
  Pair p;
  Int_t bad[kDim] = {1, 1, 1, 1, kBins[4] + 2}; // past the overflow bin of the last axis
  REQUIRE(p.h.GetBin(bad) == -1);
  Int_t neg[kDim] = {-1, 1, 1, 1, 1};
  REQUIRE(p.h.GetBin(neg) == -1);
  for (Long64_t bin : {Long64_t(-1), Long64_t(1) << 40, std::numeric_limits<Long64_t>::max()}) {
    p.h.FillBin(bin, 1.);
    p.h.SetBinContent(bin, 2.);
    p.h.AddBinContent(bin, 3.);
    p.h.SetBinError2(bin, 4.);
    REQUIRE(p.h.GetBinContent(bin) == 0.);
    REQUIRE(p.h.GetBinError2(bin) == 0.);
  }
  REQUIRE(p.h.GetNbins() == 0);

  CompactTHnSparseF empty; // as created by ROOT I/O before reading
  REQUIRE(empty.GetBinContent(Long64_t(0)) == 0.);
  REQUIRE(empty.GetNbins() == 0);
}

TEST_CASE("CompactTHnSparseToTHnSparseWideCoordinates")
{
  // 6 axes of 1030 bins: 66 bits of THnSparse coordinates (11 per axis), more than 8 bytes, yet
  // ~1.2e18 cells, which fit CompactTHnSparse's 64-bit bin index; and more bins
  // than one THnSparse chunk. The converted histogram has no hash table until its first
  // lookup; it must find every bin, survive streaming and take further fills like THnSparse.
  constexpr int nd = 6;
  Int_t nbins[nd];
  Double_t lo[nd], hi[nd];
  for (int d = 0; d < nd; ++d) {
    nbins[d] = 1030;
    lo[d] = -1.;
    hi[d] = 1.;
  }
  THnSparseD ref("ref", "ref", nd, nbins, lo, hi);
  CompactTHnSparseD h("h", "h", nd, nbins, lo, hi);
  ref.Sumw2();
  h.Sumw2();
  TRandom3 r(11);
  Double_t x[nd];
  auto fill = [&](THnBase& a, THnBase& b, int n) {
    for (int i = 0; i < n; ++i) {
      for (int d = 0; d < nd; ++d) {
        x[d] = r.Gaus(0., 0.6); // some under/overflow
      }
      Double_t w = r.Uniform(0.5, 2.);
      a.Fill(x, w);
      b.Fill(x, w);
    }
  };
  fill(ref, h, 50000);
  auto check = [&](const THnSparse& s) {
    REQUIRE(s.GetNbins() == ref.GetNbins());
    REQUIRE(s.GetEntries() == ref.GetEntries());
    REQUIRE(s.GetSumw() == Catch::Approx(ref.GetSumw()).epsilon(1e-12));
    REQUIRE(s.GetSumw2() == Catch::Approx(ref.GetSumw2()).epsilon(1e-12));
    Int_t coord[nd];
    long bad = 0;
    for (Long64_t i = 0; i < ref.GetNbins(); ++i) {
      Double_t v = ref.GetBinContent(i, coord);
      Long64_t b = s.GetBin(coord);
      bad += b < 0 || s.GetBinContent(b) != v || s.GetBinError2(b) != ref.GetBinError2(i);
    }
    REQUIRE(bad == 0);
  };
  std::unique_ptr<THnSparse> s(h.toTHnSparse());
  REQUIRE(s->IsA() == THnSparseD::Class());
  REQUIRE(s->GetNChunks() > 1);
  check(*s);

  TBufferFile buf(TBuffer::kWrite);
  buf.WriteObject(s.get());
  buf.SetReadMode();
  buf.SetBufferOffset(0);
  std::unique_ptr<THnSparse> back(static_cast<THnSparse*>(buf.ReadObject(THnSparseD::Class())));
  REQUIRE(back);
  REQUIRE(back->IsA() == THnSparseD::Class());
  check(*back);

  // further fills of the converted histogram: new bins and existing ones
  fill(ref, *s, 20000);
  check(*s);
}

TEST_CASE("CompactTHnSparseZeroWeightFills")
{
  // THnSparse keeps a bin allocated by a fill of weight 0; the conversion must too
  for (bool errors : {false, true}) {
    TRandom3 r(8);
    Pair p;
    if (errors) {
      p.ref.Sumw2();
      p.h.Sumw2();
    }
    p.fill(r, 1000, true);
    Double_t x[kDim];
    for (int i = 0; i < 100; ++i) {
      draw(r, x);
      p.ref.Fill(x, 0.);
      p.h.Fill(x, 0.);
    }
    std::unique_ptr<THnSparse> s(p.h.toTHnSparse());
    compare(p.ref, *s);
  }
}

TEST_CASE("BlockSparseArrayScatteredKeys")
{
  // Bins scattered over the whole key space, one per block (the worst case): the block
  // table must stay proportional to the bins filled, whatever the size of the key space,
  // and below THnSparse's ~70 B per filled bin (8-20 B when bins cluster)
  for (int keyBits : {30, 40, 50, 62}) {
    o2::framework::BlockSparseArray<float> a(keyBits);
    TRandom3 r(9);
    const int n = 100000;
    for (int i = 0; i < n; ++i) {
      uint64_t k = (uint64_t(r.Integer(0xffffffff)) << 32 | r.Integer(0xffffffff)) >> (64 - keyBits);
      a.add(k, 1.f);
    }
    REQUIRE(a.filledBins() > 0.99 * n);
    REQUIRE(a.bytes() < 80 * a.filledBins());
    uint64_t count = 0, last = 0;
    bool ordered = true;
    a.forEach([&](uint64_t k, float, double) { ordered = ordered && (count == 0 || k > last); last = k; ++count; });
    REQUIRE(count == a.filledBins());
    REQUIRE(ordered);
  }
}

TEST_CASE("CompactTHnSparseReadByBinNumberUnsupported")
{
  // THnBase functions built on THnSparse's numbering of filled bins throw instead of using wrong bins
  TRandom3 r(10);
  Pair p;
  p.fill(r, 1000, false);
  Double_t x[kDim];
  REQUIRE_THROWS(p.h.ComputeIntegral());
  REQUIRE_THROWS(p.h.GetRandom(x));
  REQUIRE_THROWS(p.h.PrintEntries());
  REQUIRE_THROWS(p.h.Print("c"));
  REQUIRE_THROWS(p.h.Fit(nullptr));
  REQUIRE_NOTHROW(p.h.Print("as"));
  // the same through the registry's interface type
  CompactTHnSparse& base = p.h;
  REQUIRE_THROWS(base.GetRandom(x));
  // reading the converted THnSparse works
  std::unique_ptr<THnSparse> s(p.h.toTHnSparse());
  REQUIRE(s->ComputeIntegral() > 0.);
}
