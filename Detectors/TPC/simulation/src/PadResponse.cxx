/// \file PadResponse.cxx
/// \author Andi Mathis, andreas.mathis@ph.tum.de

#include "TPCSimulation/PadResponse.h"

using namespace AliceO2::TPC;

PadResponse::PadResponse()
  : mPad(),
    mRow(),
    mWeight()
{}

PadResponse::PadResponse(Int_t pad, Int_t row, Float_t weight)
  : mPad(pad),
    mRow(row),
    mWeight(weight)
{}

PadResponse::~PadResponse(){}