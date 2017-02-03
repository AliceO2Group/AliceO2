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
  Pixel(Int_t row, Int_t col) {
    m_row = row;
    m_col = col;
  }
  Int_t GetRow() const {return m_row;};
  Int_t GetCol() const {return m_col;};
  bool operator==(const Pixel& rhs) const {
    if (m_row == rhs.m_row && m_col == rhs.m_col) {
      return true;
    }
    return false;
  }
private:
  Int_t m_row;
  Int_t m_col;
};
//////////////////////////////////////////
//////////////////////////////////////////


//////////////////////////////////////////
//////////////////////////////////////////
class Cluster {
public:
  Cluster() {
    m_rows = 0;
    m_cols = 0;
    m_pixels.clear();
  }
  void AddPixel(Int_t l, Pixel p) {
    bool found = false;
    if (m_pixels.find(l) == m_pixels.end()) m_pixels[l] = std::vector<Pixel>();
    for (size_t i = 0; i < m_pixels[l].size(); ++i) {
      if (m_pixels[l][i] == p) found = true;
    }
    if (!found) {
      m_pixels[l].push_back(p);
    } else {
      std::cout << "pixel already included in the cluster" << std::endl;
    }
  }
  Int_t GetNClusters(Int_t l) {
    if (m_pixels.find(l) == m_pixels.end()) return 0;
    return m_pixels[l].size();
  }
  Pixel GetPixel(Int_t l, Int_t digit) {
    if (m_pixels.find(l) == m_pixels.end()) return Pixel();
    return m_pixels[l][digit];
  }
  static void PixelsToClusterShape(const std::vector<Pixel>& v) {

  }
private:
  Int_t m_rows;
  Int_t m_cols;
  std::map<Int_t, std::vector<Pixel> > m_pixels;
};
//////////////////////////////////////////
//////////////////////////////////////////


//////////////////////////////////////////
//////////////////////////////////////////
void AnalyzeClusters(Int_t nev, const map<Int_t, Cluster>& clusters) {
  cout << ">>> Event " << nev << endl;
  Float_t x,z;
  for (auto const& it : clusters) {
    Cluster cls = it.second;
    cout << "id: " << it.first << " --> # digits per layer: ";
    for (auto i = 0; i < 7; ++i) {
      cout << cls.GetNClusters(i);
      if (cls.GetNClusters(i) > 0) {
        cout << " [";
        for (auto j = 0; j < cls.GetNClusters(i); ++j) {
          Pixel p = cls.GetPixel(i, j);
          //seg->detectorToLocal(p.GetRow(),p.GetCol(),x,z);
          cout << "(" << p.GetRow() << "," << p.GetCol() << ")";
          cout << "(" << x << "," << z << ")";
          if (j < cls.GetNClusters(i)-1) cout << "  ";
        }
        cout << "]";
      }
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
  TFile *file = TFile::Open("AliceO2_TGeant3.params_10.root");
  gFile->Get("FairGeoParSet");
  gman = new GeometryTGeo(kTRUE);
  seg = (SegmentationPixel*) gman->getSegmentationById(0);

  // Digits
  TFile *file1 = TFile::Open("AliceO2_TGeant3.digi_10_event.root");
  TTree *digTree=(TTree*)gFile->Get("cbmsim");
  TClonesArray digArr("AliceO2::ITS::Digit"), *pdigArr(&digArr);
  digTree->SetBranchAddress("ITSDigit",&pdigArr);


  Int_t nev=digTree->GetEntries();
  while (nev--) {
    map<Int_t, Cluster> clusters;
    digTree->GetEvent(nev);
    Int_t nd = digArr.GetEntriesFast();
    while(nd--) {
      Digit *d=(Digit *)digArr.UncheckedAt(nd);


      Int_t ix=d->getRow(), iz=d->getColumn();
      // Float_t x,z;
      // seg->detectorToLocal(ix,iz,x,z);

      Int_t chipID=d->getChipIndex();
      Int_t layer = gman->getLayer(chipID);

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
