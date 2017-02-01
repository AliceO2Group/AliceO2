/// \file PadResponse.cxx
/// \author Andi Mathis, andreas.mathis@ph.tum.de

#include "TPCSimulation/PadResponse.h"

using namespace AliceO2::TPC;

PadResponse::PadResponse()
  : mWeight(),
    mPad(),
    mRow()
{}

PadResponse::PadResponse(Int_t pad, Int_t row, Float_t weight)
  : mWeight(weight),
    mPad(pad),
    mRow(row)
{}

PadResponse::~PadResponse()= default;
