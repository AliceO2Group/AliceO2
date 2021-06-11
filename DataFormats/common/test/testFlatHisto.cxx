// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test FlatHisto class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "CommonDataFormat/FlatHisto1D.h"
#include "CommonDataFormat/FlatHisto2D.h"
#include <TFile.h>
#include <TRandom.h>
#include <TFitResult.h>
#include <TH1F.h>
#include <TH2F.h>

namespace o2
{

// basic Vertex tests
BOOST_AUTO_TEST_CASE(FlatHisto)
{
  o2::dataformats::FlatHisto1D_f h1(100, -100., 100.);
  o2::dataformats::FlatHisto2D_f h2(100, -100., 100., 50, -5., 45.);
  for (int i = 0; i < 1000000; i++) {
    h1.fill(gRandom->Gaus(10, 30));
  }
  auto h2ref = h2.createTH2F("h2ref");
  for (int i = 0; i < 10000000; i++) {
    auto x = gRandom->Gaus(10, 40), y = gRandom->Gaus(10, 10);
    h2.fill(x, y);
    h2ref->Fill(x, y);
  }
  auto th1f = h1.createTH1F();
  auto res = th1f->Fit("gaus", "S");
  BOOST_CHECK_CLOSE(res->GetParams()[1], 10, 0.2);

  printf("%e %e\n", h2.getSum(), h2ref->Integral());
  BOOST_CHECK(h2.getSum() == h2ref->Integral());

  o2::dataformats::FlatHisto1D_f h1v(h1);

  BOOST_CHECK_CLOSE(h1.getBinStart(0), -100, 1e-5);
  BOOST_CHECK_CLOSE(h1.getBinEnd(h1.getNBins() - 1), 100, 1e-5);

  BOOST_CHECK_CLOSE(h2.getBinXStart(0), -100, 1e-5);
  BOOST_CHECK_CLOSE(h2.getBinYEnd(h2.getNBinsY() - 1), 45, 1e-5);
  BOOST_CHECK_CLOSE(h2.getBinYStart(h2.getNBinsY() - 1), 45 - h2.getBinSizeY(), 1e-5);

  BOOST_CHECK(h1.canFill() && h1v.canFill());
  BOOST_CHECK(h1.getSum() == h1v.getSum());
  {
    TFile flout("flathisto.root", "recreate");
    flout.WriteObjectAny(&h1, "o2::dataformats::FlatHisto1D_f", "h1");
    flout.WriteObjectAny(&h2, "o2::dataformats::FlatHisto2D_f", "h2");
    flout.Close();
  }

  TFile flin("flathisto.root");
  o2::dataformats::FlatHisto1D_f* h1r = (o2::dataformats::FlatHisto1D_f*)flin.GetObjectUnchecked("h1");
  o2::dataformats::FlatHisto2D_f* h2r = (o2::dataformats::FlatHisto2D_f*)flin.GetObjectUnchecked("h2");
  flin.Close();
  h1r->init();
  h2r->init();

  o2::dataformats::FlatHisto1D_f h1vv;
  h1vv.adoptExternal(h1r->getView());
  BOOST_CHECK(h1.getSum() == h1vv.getSum());
  h1.add(h1vv);
  h2.subtract(*h2r);
  BOOST_CHECK(h1.getSum() == 2 * h1vv.getSum());
  BOOST_CHECK(h2.getSum() == 0.);
}

} // namespace o2
