// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef MEAN_VERTEX_DATA_H_
#define MEAN_VERTEX_DATA_H_

#include "DetectorsCalibration/TimeSlot.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include <array>
#include <deque>
#include <gsl/span>

namespace o2
{
namespace calibration
{

struct MeanVertexData {

  using PVertex = o2::dataformats::PrimaryVertex;
  float rangeX = 10.f;
  float rangeY = 10.f;
  float rangeZ = 10.f;
  int nbinsX = 1000;
  int nbinsY = 1000;
  int nbinsZ = 1000;
  float v2BinX = nbinsX / (2 * rangeX);
  float v2BinY = nbinsY / (2 * rangeY);
  float v2BinZ = nbinsZ / (2 * rangeZ);
  int entries = 0;
  std::deque<float> histoX{0};
  std::deque<float> histoY{0};
  std::deque<float> histoZ{0};
  bool useFit = false;
  
  MeanVertexData();

  ~MeanVertexData() {
    useFit = false;
    nbinsX = 1000;
    rangeX = 10.f;
    nbinsY = 1000;
    rangeY = 10.f;
    nbinsZ = 1000;
    rangeZ = 10.f;
    v2BinX = 0.f;
    v2BinY = 0.f;
    v2BinZ = 0.f;
    histoX.clear();
    histoY.clear();
    histoZ.clear();
  }

MeanVertexData(bool buseFit, int nbX, float rX, int nbY, float rY, int nbZ, float rZ) :
  useFit(buseFit), nbinsX(nbX), rangeX(rX), v2BinX(0), nbinsY(nbY), rangeY(rY), v2BinY(0),
    nbinsZ(nbZ), rangeZ(rZ), v2BinZ(0) 
  {
    //    if (useFit) {
      if (rX <= 0. || nbX < 1 || rY <= 0. || nbY < 1 || rZ <= 0. || nbZ < 1) {
	throw std::runtime_error("Wrong initialization of the histogram");
      }
      v2BinX = nbinsX / (2 * rangeX);
      v2BinY = nbinsY / (2 * rangeY);
      v2BinZ = nbinsZ / (2 * rangeZ);
      histoX.resize(nbinsX, 0.);
      histoY.resize(nbinsY, 0.);
      histoZ.resize(nbinsZ, 0.);                           
    }
  //}

//_____________________________________________
MeanVertexData(MeanVertexData&& other) {

    // move constructor
    
    useFit = other.useFit;
    nbinsX = other.nbinsX;
    rangeX = other.rangeX;
    nbinsY = other.nbinsY;
    rangeY = other.rangeY;
    nbinsZ = other.nbinsZ;
    rangeZ = other.rangeZ;
    v2BinX = other.v2BinX;
    v2BinY = other.v2BinY;
    v2BinZ = other.v2BinZ;
    histoX = other.histoX;
    histoY = other.histoY;
    histoZ = other.histoZ;
    other.useFit = false;
    other.nbinsX = 1000;
    other.rangeX = 10.f;
    other.nbinsY = 1000;
    other.rangeY = 10.f;
    other.nbinsZ = 1000;
    other.rangeZ = 10.f;
    other.v2BinX = nbinsX / (2 * rangeX);
    other.v2BinY = nbinsY / (2 * rangeY);
    other.v2BinZ = nbinsZ / (2 * rangeZ);
    other.histoX.clear();
    other.histoY.clear();
    other.histoZ.clear();
}

//_____________________________________________
MeanVertexData(const MeanVertexData& other) {

    // copy constructor
    
    useFit = other.useFit;
    nbinsX = other.nbinsX;
    rangeX = other.rangeX;
    nbinsY = other.nbinsY;
    rangeY = other.rangeY;
    nbinsZ = other.nbinsZ;
    rangeZ = other.rangeZ;
    v2BinX = other.v2BinX;
    v2BinY = other.v2BinY;
    v2BinZ = other.v2BinZ;
    histoX = other.histoX;
    histoY = other.histoY;
    histoZ = other.histoZ;
}

//_____________________________________________
MeanVertexData& operator = (MeanVertexData& other) {

    // assignment operator
    
    useFit = other.useFit;
    nbinsX = other.nbinsX;
    rangeX = other.rangeX;
    nbinsY = other.nbinsY;
    rangeY = other.rangeY;
    nbinsZ = other.nbinsZ;
    rangeZ = other.rangeZ;
    v2BinX = other.v2BinX;
    v2BinY = other.v2BinY;
    v2BinZ = other.v2BinZ;
    histoX = other.histoX;
    histoY = other.histoY;
    histoZ = other.histoZ;
    other.useFit = false;
    other.nbinsX = 1000;
    other.rangeX = 10.f;
    other.nbinsY = 1000;
    other.rangeY = 10.f;
    other.nbinsZ = 1000;
    other.rangeZ = 10.f;
    other.v2BinX = nbinsX / (2 * rangeX);
    other.v2BinY = nbinsY / (2 * rangeY);
    other.v2BinZ = nbinsZ / (2 * rangeZ);
    other.histoX.clear();
    other.histoY.clear();
    other.histoZ.clear();
    return *this;
}

//_____________________________________________
MeanVertexData& operator = (MeanVertexData&& other) {

    // move assignment operator
    
    useFit = other.useFit;
    nbinsX = other.nbinsX;
    rangeX = other.rangeX;
    nbinsY = other.nbinsY;
    rangeY = other.rangeY;
    nbinsZ = other.nbinsZ;
    rangeZ = other.rangeZ;
    v2BinX = other.v2BinX;
    v2BinY = other.v2BinY;
    v2BinZ = other.v2BinZ;
    histoX = other.histoX;
    histoY = other.histoY;
    histoZ = other.histoZ;
    other.useFit = false;
    other.nbinsX = 1000;
    other.rangeX = 10.f;
    other.nbinsY = 1000;
    other.rangeY = 10.f;
    other.nbinsZ = 1000;
    other.rangeZ = 10.f;
    other.v2BinX = nbinsX / (2 * rangeX);
    other.v2BinY = nbinsY / (2 * rangeY);
    other.v2BinZ = nbinsZ / (2 * rangeZ);
    other.histoX.clear();
    other.histoY.clear();
    other.histoZ.clear();
    return *this;
}

//_____________________________________________
void init(bool buseFit, int nbX, float rX, int nbY, float rY, int nbZ, float rZ) {

  useFit = buseFit;
  nbinsX = nbX;
  rangeX = rX;
  nbinsY = nbY;
  rangeY = rY; 
  nbinsZ = nbZ;
  rangeZ = rZ;
  v2BinX = nbinsX / (2 * rangeX);
  v2BinY = nbinsY / (2 * rangeY);
  v2BinZ = nbinsZ / (2 * rangeZ);
  histoX.resize(nbinsX, 0.);
  histoY.resize(nbinsY, 0.);
  histoZ.resize(nbinsZ, 0.);  
}

//_____________________________________________

  size_t getEntries() const { return entries; }
  void print() const;
  void fill(const gsl::span<const PVertex> data);
  void merge(const MeanVertexData* prev);
  void subtract(const MeanVertexData* prev);

  ClassDefNV(MeanVertexData, 1);
};

} // end namespace calibration
} // end namespace o2

#endif /* MEAN_VERTEX_DATA_H_ */
