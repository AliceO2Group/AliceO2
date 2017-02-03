/// \file CheckDigits.C
/// \brief Simple macro to check ITSU digits

#if !defined(__CINT__) || defined(__MAKECINT__)
#include <TFile.h>
#include <TTree.h>
#include <TClonesArray.h>
#include <TH2F.h>
#include <TNtuple.h>
#include <TCanvas.h>

#include "ITSBase/GeometryTGeo.h"
#include "ITSBase/SegmentationPixel.h"
#include "ITSBase/Digit.h"
#include "ITSSimulation/Point.h"
#include "ITSSimulation/ClusterShape.h"
#endif

using namespace AliceO2::ITS;
GeometryTGeo *gman;
SegmentationPixel *seg;

//////////////////////////////////////////
//////////////////////////////////////////
class Pixel {
public:
  Pixel() {
    m_row = 0;
    m_col = 0;
  }
  Pixel(UInt_t row, UInt_t col) {
    m_row = row;
    m_col = col;
  }
  UInt_t GetRow() const {return m_row;};
  UInt_t GetCol() const {return m_col;};
  bool operator==(const Pixel& rhs) const {
    if (m_row == rhs.m_row && m_col == rhs.m_col) {
      return true;
    }
    return false;
  }
private:
  UInt_t m_row;
  UInt_t m_col;
};
//////////////////////////////////////////
//////////////////////////////////////////


//////////////////////////////////////////
//////////////////////////////////////////
class Cluster {
public:
  Cluster() {
    m_pixels.clear();
  }
  void AddPixel(UInt_t l, Pixel p) {
    bool found = false;
    if (m_pixels.find(l) == m_pixels.end()) m_pixels[l] = std::vector<Pixel>();
    for (size_t i = 0; i < m_pixels[l].size(); ++i) {
      if (m_pixels[l][i] == p) found = true;
    }
    if (!found) {
      m_pixels[l].push_back(p);
    } else {
      std::cout << "l: " << l << " - d:" << "(" << p.GetRow() << ", " << p.GetCol() << ") - already in cluster!" << std::endl;
    }
  }
  UInt_t GetNClusters(UInt_t l) {
    if (m_pixels.find(l) == m_pixels.end()) return 0;
    return m_pixels[l].size();
  }
  Pixel GetPixel(UInt_t l, UInt_t digit) {
    if (m_pixels.find(l) == m_pixels.end()) return Pixel();
    return m_pixels[l][digit];
  }
  const vector<Pixel>& GetPixels(UInt_t l) {
    return m_pixels[l];
  }

  static ClusterShape* PixelsToClusterShape(const std::vector<Pixel>& v) {
    if (v.size() == 0) return new ClusterShape();
    vector<UInt_t> r;
    vector<UInt_t> c;
    UInt_t min_r = UINT_MAX, min_c = UINT_MAX;
    for (auto & p : v) {
      if (p.GetRow() < min_r) min_r = p.GetRow();
      if (p.GetCol() < min_c) min_c = p.GetCol();
      if (find(begin(r), end(r), p.GetRow()) == end(r)) r.push_back(p.GetRow());
      if (find(begin(c), end(c), p.GetCol()) == end(c)) c.push_back(p.GetCol());
    }

    UInt_t rows = r.size();
    UInt_t cols = c.size();
    std::vector<UInt_t> pindex;
    for (auto & p : v) {
      Int_t rd = p.GetRow() - min_r;
      Int_t cd = p.GetCol() - min_c;
      Int_t index = rd * cols + cd;
      pindex.push_back(index);
    }

    return new ClusterShape(rows, cols, pindex);
  }
private:
  map<UInt_t, vector<Pixel> > m_pixels;
};
//////////////////////////////////////////
//////////////////////////////////////////


//////////////////////////////////////////
//////////////////////////////////////////
void AnalyzeClusters(Int_t nev, const map<UInt_t, Cluster>& clusters) {
  cout << ">>> Event " << nev << endl;
  Float_t x,z;
  for (auto const& it : clusters) {
    Cluster cls = it.second;
    cout << "id: " << it.first << " Ndigit: ";
    for (auto i = 0; i < 7; ++i) {
      cout << cls.GetNClusters(i);
      vector<Pixel> pv = cls.GetPixels(i);
      ClusterShape *cs = Cluster::PixelsToClusterShape(pv);

      cout << endl << endl << *cs << endl << endl;

      // if (cls.GetNClusters(i) > 0) {
      //   cout << " [";
      //   for (auto j = 0; j < cls.GetNClusters(i); ++j) {
      //     Pixel p = cls.GetPixel(i, j);
      //     cout << "(" << p.GetRow() << "," << p.GetCol() << ")";
      //     //seg->detectorToLocal(p.GetRow(),p.GetCol(),x,z);
      //     //cout << "(" << x << "," << z << ")";
      //     if (j < cls.GetNClusters(i)-1) cout << "  ";
      //   }
      //   cout << "]";
      // }

      if (i < 6) cout << ", ";
      else cout << endl;
    }
}
  cout << ">>>>>>>>>>>>>>>>>" << endl << endl;
}
//////////////////////////////////////////
//////////////////////////////////////////


void CheckClusterShape() {
  TFile *f=TFile::Open("Shapes.root","recreate");

  // Geometry
  TFile *file = TFile::Open("AliceO2_TGeant3.params.root");
  gFile->Get("FairGeoParSet");
  gman = new GeometryTGeo(kTRUE);
  seg = (SegmentationPixel*) gman->getSegmentationById(0);

  // Digits
  TFile *file1 = TFile::Open("AliceO2_TGeant3.digi.root");
  TTree *digTree=(TTree*)gFile->Get("cbmsim");
  TClonesArray digArr("AliceO2::ITS::Digit"), *pdigArr(&digArr);
  digTree->SetBranchAddress("ITSDigit",&pdigArr);

  Int_t nev=digTree->GetEntries();
  while (nev--) {
    map<UInt_t, Cluster> clusters;
    digTree->GetEvent(nev);
    Int_t nd = digArr.GetEntriesFast();
    while(nd--) {
      Digit *d=(Digit *)digArr.UncheckedAt(nd);
      Int_t ix=d->getRow(), iz=d->getColumn();

      Int_t chipID=d->getChipIndex();
      UInt_t layer = gman->getLayer(chipID);

      Pixel pixels(ix, iz);
      if (clusters.find(d->getLabel(0)) == clusters.end()) {
        Cluster c;
        clusters[d->getLabel(0)] = c;
      }
      clusters[d->getLabel(0)].AddPixel(layer, pixels);
    }
    AnalyzeClusters(nev, clusters);
  }

  f->Write();
  f->Close();
}
