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

/// \file field.geom
/// \author David Rohr
/// \brief Magnetic field lines: integrates the field from each seed point (input point) and emits the line strip

#version 460 core
#extension GL_GOOGLE_include_directive : require
#include "../helpers/GPUDisplayMagneticFieldConstants.h"

#define DIMENSIONS GPUCA_DISPLAY_FIELD_DIMENSIONS
#define SOL_Z_SEGS GPUCA_DISPLAY_FIELD_MAX_SOLENOID_Z_SEGMENTS
#define SOL_P_SEGS GPUCA_DISPLAY_FIELD_MAX_SOLENOID_P_SEGMENTS
#define SOL_R_SEGS GPUCA_DISPLAY_FIELD_MAX_SOLENOID_R_SEGMENTS
#define SOL_PARAMS GPUCA_DISPLAY_FIELD_MAX_SOLENOID_PARAMETERIZATIONS
#define SOL_ROWS GPUCA_DISPLAY_FIELD_MAX_SOLENOID_ROWS
#define SOL_COLUMNS GPUCA_DISPLAY_FIELD_MAX_SOLENOID_COLUMNS
#define SOL_COEFFS GPUCA_DISPLAY_FIELD_MAX_SOLENOID_COEFFICIENTS
#define DIP_Z_SEGS GPUCA_DISPLAY_FIELD_MAX_DIPOLE_Z_SEGMENTS
#define DIP_Y_SEGS GPUCA_DISPLAY_FIELD_MAX_DIPOLE_Y_SEGMENTS
#define DIP_X_SEGS GPUCA_DISPLAY_FIELD_MAX_DIPOLE_X_SEGMENTS
#define DIP_PARAMS GPUCA_DISPLAY_FIELD_MAX_DIPOLE_PARAMETERIZATIONS
#define DIP_ROWS GPUCA_DISPLAY_FIELD_MAX_DIPOLE_ROWS
#define DIP_COLUMNS GPUCA_DISPLAY_FIELD_MAX_DIPOLE_COLUMNS
#define DIP_COEFFS GPUCA_DISPLAY_FIELD_MAX_DIPOLE_COEFFICIENTS
#define MAX_CHEB_ORDER GPUCA_DISPLAY_FIELD_MAX_CHEBYSHEV_ORDER

#ifdef VULKAN
#define FIELD_SSBO_SET set = 1, // Vulkan: UBO and SSBOs share the binding numbers of a descriptor set
#else
#define FIELD_SSBO_SET
#endif

layout(std430, FIELD_SSBO_SET binding = 0) restrict readonly buffer field_config_ssbo {
    uint StepCount;
    float StepSize;
} field_config;

layout(std430, FIELD_SSBO_SET binding = 1) restrict readonly buffer sol_segment_ssbo {
    float MinZ;
    float MaxZ;
    float MultiplicativeFactor;

    int ZSegments;

    float SegZSol[SOL_Z_SEGS];

    int BegSegPSol[SOL_Z_SEGS];
    int NSegPSol[SOL_Z_SEGS];

    float SegPSol[SOL_P_SEGS];
    int BegSegRSol[SOL_P_SEGS];
    int NSegRSol[SOL_P_SEGS];

    float SegRSol[SOL_R_SEGS];
    int SegIDSol[SOL_R_SEGS];
} sol_segment;

layout(std430, FIELD_SSBO_SET binding = 2) restrict readonly buffer dip_segment_ssbo {
    float MinZ;
    float MaxZ;
    float MultiplicativeFactor;

    int ZSegments;

    float SegZDip[DIP_Z_SEGS];

    int BegSegYDip[DIP_Z_SEGS];
    int NSegYDip[DIP_Z_SEGS];

    float SegYDip[DIP_Y_SEGS];
    int BegSegXDip[DIP_Y_SEGS];
    int NSegXDip[DIP_Y_SEGS];

    float SegXDip[DIP_X_SEGS];
    int SegIDDip[DIP_X_SEGS];
} dip_segment;

layout(std430, FIELD_SSBO_SET binding = 3) restrict readonly buffer sol_params_ssbo {
    float BOffsets[DIMENSIONS*SOL_PARAMS];
    float BScales[DIMENSIONS*SOL_PARAMS];
    float BMin[DIMENSIONS*SOL_PARAMS];
    float BMax[DIMENSIONS*SOL_PARAMS];
    int NRows[DIMENSIONS*SOL_PARAMS];
    int ColsAtRowOffset[DIMENSIONS*SOL_PARAMS];
    int CofsAtRowOffset[DIMENSIONS*SOL_PARAMS];

    int NColsAtRow[SOL_ROWS];
    int CofsAtColOffset[SOL_ROWS];

    int NCofsAtCol[SOL_COLUMNS];
    int AtColCoefOffset[SOL_COLUMNS];

    float Coeffs[SOL_COEFFS];
} sol_params;

layout(std430, FIELD_SSBO_SET binding = 4) restrict readonly buffer dip_params_ssbo {
    float BOffsets[DIMENSIONS*DIP_PARAMS];
    float BScales[DIMENSIONS*DIP_PARAMS];
    float BMin[DIMENSIONS*DIP_PARAMS];
    float BMax[DIMENSIONS*DIP_PARAMS];
    int NRows[DIMENSIONS*DIP_PARAMS];
    int ColsAtRowOffset[DIMENSIONS*DIP_PARAMS];
    int CofsAtRowOffset[DIMENSIONS*DIP_PARAMS];

    int NColsAtRow[DIP_ROWS];
    int CofsAtColOffset[DIP_ROWS];

    int NCofsAtCol[DIP_COLUMNS];
    int AtColCoefOffset[DIP_COLUMNS];

    float Coeffs[DIP_COEFFS];
} dip_params;

float tmpCfs1[MAX_CHEB_ORDER];
float tmpCfs0[MAX_CHEB_ORDER];

vec3 CarttoCyl(vec3 pos) {
    return vec3(length(pos.xy), atan(pos.y, pos.x), pos.z);
}

int findSolSegment(vec3 pos) {
    int rid,pid,zid;

    for(zid=0; zid < sol_segment.ZSegments; zid++) if(pos.z<sol_segment.SegZSol[zid]) break;
    if(--zid < 0) zid = 0;

    const int psegBeg = sol_segment.BegSegPSol[zid];
    for(pid=0; pid<sol_segment.NSegPSol[zid]; pid++) if(pos.y<sol_segment.SegPSol[psegBeg+pid]) break;
    if(--pid < 0) pid = 0;
    pid += psegBeg;

    const int rsegBeg = sol_segment.BegSegRSol[pid];
    for(rid=0; rid<sol_segment.NSegRSol[pid]; rid++) if(pos.x<sol_segment.SegRSol[rsegBeg+rid]) break;
    if(--rid < 0) rid = 0;
    rid += rsegBeg;

    return sol_segment.SegIDSol[rid];
}

int findDipSegment(vec3 pos) {
    int xid,yid,zid;

    for(zid=0; zid < dip_segment.ZSegments; zid++) if(pos.z<dip_segment.SegZDip[zid]) break;
    if(--zid < 0) zid = 0;

    const int ysegBeg = dip_segment.BegSegYDip[zid];
    for(yid=0; yid<dip_segment.NSegYDip[zid]; yid++) if(pos.y<dip_segment.SegYDip[ysegBeg+yid]) break;
    if(--yid < 0) yid = 0;
    yid += ysegBeg;

    const int xsegBeg = dip_segment.BegSegXDip[yid];
    for(xid=0; xid<dip_segment.NSegXDip[yid]; xid++) if(pos.x<dip_segment.SegXDip[xsegBeg+xid]) break;
    if(--xid < 0) xid = 0;
    xid += xsegBeg;

    return dip_segment.SegIDDip[xid];
}

vec3 mapToInternalSol(int segID, vec3 rphiz) {
    const int index = DIMENSIONS*segID;
    vec3 offsets = vec3(sol_params.BOffsets[index+0], sol_params.BOffsets[index+1], sol_params.BOffsets[index+2]);
    vec3 scales = vec3(sol_params.BScales[index+0], sol_params.BScales[index+1], sol_params.BScales[index+2]);

    return (rphiz-offsets)*scales;
}

vec3 mapToInternalDip(int segID, vec3 pos) {
    const int index = DIMENSIONS*segID;
    const vec3 offsets = vec3(dip_params.BOffsets[index+0], dip_params.BOffsets[index+1], dip_params.BOffsets[index+2]);
    const vec3 scales = vec3(dip_params.BScales[index+0], dip_params.BScales[index+1], dip_params.BScales[index+2]);

    return (pos-offsets)*scales;
}

float cheb1DArray(float x, float arr[MAX_CHEB_ORDER], int ncf) {
    if(ncf <= 0) return 0.0f;

    const float x2 = 2*x;

    vec3 b = vec3(arr[--ncf], 0, 0);
    --ncf;

    const vec3 t1 = vec3(1, x2, -1);

    for (int i=ncf;i>=0;i--) {
        b.zy = b.yx;
        b.x = arr[i];
        b.x = dot(t1, b);
    }

    const vec3 t = vec3(1, -x, 0);
    return dot(t, b);
}

float cheb1DParamsSol(float x, int coeff_offset, int ncf) {
    if(ncf <= 0) return 0.0f;

    const float x2 = 2*x;

    vec3 b = vec3(sol_params.Coeffs[coeff_offset + (--ncf)], 0, 0);
    --ncf;

    const vec3 t1 = vec3(1, x2, -1);

    for (int i=ncf;i>=0;i--) {
        b.zy = b.yx;
        b.x = sol_params.Coeffs[coeff_offset + i];
        b.x = dot(t1, b);
    }

    const vec3 t = vec3(1, -x, 0);
    return dot(t, b);
}

float cheb1DParamsDip(float x, int coeff_offset, int ncf) {
    if(ncf <= 0) return 0.0f;

    const float x2 = 2*x;

    vec3 b = vec3(dip_params.Coeffs[coeff_offset + (--ncf)], 0, 0);
    --ncf;

    const vec3 t1 = vec3(1, x2, -1);

    for (int i=ncf;i>=0;i--) {
        b.zy = b.yx;
        b.x = dip_params.Coeffs[coeff_offset + i];
        b.x = dot(t1, b);
    }

    const vec3 t = vec3(1, -x, 0);
    return dot(t, b);
}

bool IsBetween(vec3 sMin, vec3 val, vec3 sMax) {
    return all(lessThanEqual(sMin, val)) && all(lessThanEqual(val, sMax));
}

bool IsInsideSol(int segID, vec3 rphiz) {
    const int index = DIMENSIONS*segID;

    const vec3 seg_min = vec3(sol_params.BMin[index+0], sol_params.BMin[index+1], sol_params.BMin[index+2]);
    const vec3 seg_max = vec3(sol_params.BMax[index+0], sol_params.BMax[index+1], sol_params.BMax[index+2]);

    return IsBetween(seg_min, rphiz, seg_max);
}

bool IsInsideDip(int segID, vec3 pos) {
    const int index = DIMENSIONS*segID;

    const vec3 seg_min = vec3(dip_params.BMin[index+0], dip_params.BMin[index+1], dip_params.BMin[index+2]);
    const vec3 seg_max = vec3(dip_params.BMax[index+0], dip_params.BMax[index+1], dip_params.BMax[index+2]);

    return IsBetween(seg_min, pos, seg_max);
}

float Eval3DSol(int segID, int dim, vec3 internal) {
    const int index = DIMENSIONS*segID;
    const int n_rows = sol_params.NRows[index+dim];
    const int cols_at_row_offset = sol_params.ColsAtRowOffset[index+dim];
    const int coeffs_at_row_offset = sol_params.CofsAtRowOffset[index+dim];

    for(int row = 0; row < n_rows; row++) {
        const int n_cols = sol_params.NColsAtRow[cols_at_row_offset+row];
        const int coeff_at_col_offset = sol_params.CofsAtColOffset[cols_at_row_offset+row];

        for(int col = 0; col < n_cols; col++) {
            const int n_coeffs = sol_params.NCofsAtCol[coeff_at_col_offset+col];
            const int per_col_coeff_offset = sol_params.AtColCoefOffset[coeff_at_col_offset+col];

            const int coeffs_offset = coeffs_at_row_offset + per_col_coeff_offset;

            tmpCfs1[col] = cheb1DParamsSol(internal.z, coeffs_offset,n_coeffs);
        }
        tmpCfs0[row] = cheb1DArray(internal.y, tmpCfs1, n_cols);
    }

    return cheb1DArray(internal.x, tmpCfs0, n_rows);
}

vec3 EvalSol(int segID, vec3 rphiz) {
    const vec3 internal = mapToInternalSol(segID, rphiz);
    return vec3(Eval3DSol(segID, 0, internal), Eval3DSol(segID, 1, internal), Eval3DSol(segID, 2, internal));
}

float Eval3DDip(int segID, int dim, vec3 internal) {
    const int index = DIMENSIONS*segID;
    const int n_rows = dip_params.NRows[index+dim];
    const int cols_at_row_offset = dip_params.ColsAtRowOffset[index+dim];
    const int coeffs_at_row_offset = dip_params.CofsAtRowOffset[index+dim];

    for(int row = 0; row < n_rows; row++) {
        const int n_cols = dip_params.NColsAtRow[cols_at_row_offset+row];
        const int coeff_at_col_offset = dip_params.CofsAtColOffset[cols_at_row_offset+row];

        for(int col = 0; col < n_cols; col++) {
            const int n_coeffs = dip_params.NCofsAtCol[coeff_at_col_offset+col];
            const int per_col_coeff_offset = dip_params.AtColCoefOffset[coeff_at_col_offset+col];

            const int coeffs_offset = coeffs_at_row_offset + per_col_coeff_offset;

            tmpCfs1[col] = cheb1DParamsDip(internal.z, coeffs_offset, n_coeffs);
        }
        tmpCfs0[row] = cheb1DArray(internal.y, tmpCfs1, n_cols);
    }

    return cheb1DArray(internal.x, tmpCfs0, n_rows);
}

vec3 EvalDip(int segID, vec3 pos) {
    const vec3 internal = mapToInternalDip(segID, pos);
    return vec3(Eval3DDip(segID, 0, internal), Eval3DDip(segID, 1, internal), Eval3DDip(segID, 2, internal));
}

vec3 CyltoCartCylB(vec3 rphiz, vec3 brphiz) {
    const float btr = length(brphiz.xy);
    const float psiPLUSphi = atan(brphiz.y, brphiz.x) + rphiz.y;

    return vec3(btr*cos(psiPLUSphi), btr*sin(psiPLUSphi), brphiz.z);
}

vec3 MachineField(vec3 pos) {
    return vec3(0);
}

vec3 SolDipField(vec3 pos) {
    if(pos.z > sol_segment.MinZ) {
        const vec3 rphiz = CarttoCyl(pos);
        const int segID = findSolSegment(rphiz);
        if(segID >=0 && IsInsideSol(segID, rphiz)) {
            const vec3 brphiz = EvalSol(segID, rphiz);
            return CyltoCartCylB(rphiz, brphiz) * sol_segment.MultiplicativeFactor;
        }
    }

    const int segID = findDipSegment(pos);
    if(segID >= 0 && IsInsideDip(segID, pos)) {
        return EvalDip(segID, pos) * dip_segment.MultiplicativeFactor;
    }

    return vec3(0);
}

vec3 Field(vec3 pos) {
    if(pos.z > dip_segment.MinZ && pos.z < sol_segment.MaxZ) {
        return SolDipField(pos);
    }
    return vec3(0);
}

layout (points) in;
layout (line_strip, max_vertices = 256) out;

layout (binding = 0) uniform uniformMatrix { mat4 ModelViewProj; } um;

const float positionScale = 100.0f;

void main() {
    vec3 position = gl_in[0].gl_Position.xyz;

    for(uint i = 0; i < field_config.StepCount; i++) {
        gl_Position = um.ModelViewProj * vec4(position/positionScale, 1.0f);
        EmitVertex();
        const vec3 b_vec = Field(position);
        position -= b_vec * field_config.StepSize;
    }
    EndPrimitive();
}
