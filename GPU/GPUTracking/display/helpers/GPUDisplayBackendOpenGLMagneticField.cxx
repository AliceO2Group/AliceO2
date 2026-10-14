// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayBackendOpenGLMagneticField.cxx
/// \author Piotr Nowakowski

#ifdef GPUCA_DISPLAY_GL3W
#include "GL/gl3w.h"
#else
#include <GL/glew.h>
#endif
#if __has_include(<GL/glu.h>)
#include <GL/glu.h>
#else
#define gluErrorString(err) ""
#endif

#include "GPUCommonDef.h"
#include "GPUDisplayMagneticField.h"
#include "backend/GPUDisplayBackendOpenGL.h"
#include "GPUDisplay.h"
#include "GPULogging.h"

using namespace o2::gpu;

#ifdef GPUCA_BUILD_EVENT_DISPLAY_OPENGL // Set by CMake, see GPUDisplayBackendOpenGL.cxx

#define GPUCA_DISPLAY_SHADER_TYPE_OPENGL
#include "GPUDisplayLoadShaders.h"

#define CHKERR(cmd)                                                                                                 \
  do {                                                                                                              \
    (cmd);                                                                                                          \
    GLenum err = glGetError();                                                                                      \
    while (err != GL_NO_ERROR) {                                                                                    \
      GPUError("OpenGL Error %d: %s (%s: %d)", (int32_t)err, (const char*)gluErrorString(err), __FILE__, __LINE__); \
      throw std::runtime_error("OpenGL Failure");                                                                   \
    }                                                                                                               \
  } while (false)

int32_t GPUDisplayBackendOpenGL::InitMagFieldVisualization()
{
  mMagneticFieldVisualization = std::make_unique<GPUDisplayMagneticField>();
  if (!mMagneticFieldVisualization->mValid) {
    GPUError("Magnetic field data not available, field drawing disabled");
    mMagneticFieldVisualization.reset();
    return 1;
  }
  mMagneticFieldVisualization->generateSeedPoints(mDisplay->cfgL().bFieldLinesCount);

  CHKERR(mVertexShaderPassthrough = loadShader(GL_VERTEX_SHADER, _binary_opengl_vertexPassthrough_vert_spv_start, _binary_opengl_vertexPassthrough_vert_spv_len));
  CHKERR(mGeometryShader = loadShader(GL_GEOMETRY_SHADER, _binary_opengl_field_geom_spv_start, _binary_opengl_field_geom_spv_len));

  if (checkShaderStatus(mVertexShaderPassthrough) || checkShaderStatus(mGeometryShader)) {
    mMagneticFieldVisualization.reset();
    return 1;
  }

  CHKERR(mShaderProgramField = glCreateProgram());
  CHKERR(glAttachShader(mShaderProgramField, mVertexShaderPassthrough));
  CHKERR(glAttachShader(mShaderProgramField, mGeometryShader));
  CHKERR(glAttachShader(mShaderProgramField, mFragmentShader));
  CHKERR(glLinkProgram(mShaderProgramField));

  if (checkProgramStatus(mShaderProgramField)) {
    mMagneticFieldVisualization.reset();
    return 1;
  }

  const auto ATTRIB_ZERO = 0;
  const auto BUFFER_IDX = 0;

  CHKERR(glCreateVertexArrays(1, &VAO_field));
  CHKERR(glEnableVertexArrayAttrib(VAO_field, ATTRIB_ZERO));
  CHKERR(glVertexArrayAttribFormat(VAO_field, ATTRIB_ZERO, 3, GL_FLOAT, GL_FALSE, 0));

  CHKERR(glCreateBuffers(1, &VBO_field));
  CHKERR(glNamedBufferData(VBO_field, mMagneticFieldVisualization->mFieldLineSeedPoints.size() * sizeof(GPUDisplayMagneticField::vtx), mMagneticFieldVisualization->mFieldLineSeedPoints.data(), GL_STATIC_DRAW));

  CHKERR(glVertexArrayVertexBuffer(VAO_field, BUFFER_IDX, VBO_field, 0, sizeof(GPUDisplayMagneticField::vtx)));

  CHKERR(glVertexArrayAttribBinding(VAO_field, ATTRIB_ZERO, BUFFER_IDX));

  CHKERR(glCreateBuffers(1, &mFieldModelViewBuffer));
  CHKERR(glNamedBufferData(mFieldModelViewBuffer, sizeof(hmm_mat4), nullptr, GL_STREAM_DRAW));

  CHKERR(glCreateBuffers(1, &mFieldModelConstantsBuffer));
  CHKERR(glNamedBufferData(mFieldModelConstantsBuffer, sizeof(GPUDisplayMagneticField::RenderConstantsUniform), mMagneticFieldVisualization->mRenderConstantsUniform.get(), GL_STREAM_DRAW));

  CHKERR(glCreateBuffers(1, &mSolenoidSegmentsBuffer));
  CHKERR(glNamedBufferData(mSolenoidSegmentsBuffer, sizeof(GPUDisplayMagneticField::SolenoidSegmentsUniform), mMagneticFieldVisualization->mSolenoidSegments.get(), GL_STREAM_DRAW));
  CHKERR(glCreateBuffers(1, &mSolenoidParameterizationBuffer));
  CHKERR(glNamedBufferData(mSolenoidParameterizationBuffer, sizeof(GPUDisplayMagneticField::SolenoidParameterizationUniform), mMagneticFieldVisualization->mSolenoidParameterization.get(), GL_STREAM_DRAW));

  CHKERR(glCreateBuffers(1, &mDipoleSegmentsBuffer));
  CHKERR(glNamedBufferData(mDipoleSegmentsBuffer, sizeof(GPUDisplayMagneticField::DipoleSegmentsUniform), mMagneticFieldVisualization->mDipoleSegments.get(), GL_STREAM_DRAW));
  CHKERR(glCreateBuffers(1, &mDipoleParameterizationBuffer));
  CHKERR(glNamedBufferData(mDipoleParameterizationBuffer, sizeof(GPUDisplayMagneticField::DipoleParameterizationUniform), mMagneticFieldVisualization->mDipoleParameterization.get(), GL_STREAM_DRAW));

  return 0;
}

uint32_t GPUDisplayBackendOpenGL::drawField()
{
  if (!mMagneticFieldVisualization) {
    if (!mMagneticFieldFailed && InitMagFieldVisualization()) {
      mMagneticFieldFailed = true;
    }
    return 0; // next frame will fill MVP matrix
  }

  if (mMagneticFieldVisualization->mFieldLineSeedPoints.size() != (uint32_t)mDisplay->cfgL().bFieldLinesCount) {
    mMagneticFieldVisualization->generateSeedPoints(mDisplay->cfgL().bFieldLinesCount);
    CHKERR(glNamedBufferData(VBO_field, mMagneticFieldVisualization->mFieldLineSeedPoints.size() * sizeof(GPUDisplayMagneticField::vtx), mMagneticFieldVisualization->mFieldLineSeedPoints.data(), GL_STATIC_DRAW));
  }

  mMagneticFieldVisualization->mRenderConstantsUniform->StepSize = mDisplay->cfgL().bFieldStepSize;
  mMagneticFieldVisualization->mRenderConstantsUniform->StepCount = mDisplay->cfgL().bFieldStepCount;
  CHKERR(glNamedBufferSubData(mFieldModelConstantsBuffer, 0, sizeof(GPUDisplayMagneticField::RenderConstantsUniform), mMagneticFieldVisualization->mRenderConstantsUniform.get()));

  CHKERR(glBindVertexArray(VAO_field));
  CHKERR(glUseProgram(mShaderProgramField));
  CHKERR(glBindBufferBase(GL_UNIFORM_BUFFER, 0, mFieldModelViewBuffer));
  const std::array<float, 4> drawColor = {1.f, 0.f, 0.f, 1.f};
  CHKERR(glUniform4fv(UNIFORM_LOCATION_COLOR, 1, drawColor.data()));

  CHKERR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, mFieldModelConstantsBuffer));
  CHKERR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mSolenoidSegmentsBuffer));
  CHKERR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, mDipoleSegmentsBuffer));
  CHKERR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, mSolenoidParameterizationBuffer));
  CHKERR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, mDipoleParameterizationBuffer));

  CHKERR(glDrawArrays(GL_POINTS, 0, mMagneticFieldVisualization->mFieldLineSeedPoints.size()));

  CHKERR(glUseProgram(0));
  CHKERR(glBindVertexArray(0));

  return 0;
}

void GPUDisplayBackendOpenGL::ExitMagFieldVisualization()
{
  CHKERR(glDeleteBuffers(1, &mFieldModelViewBuffer));
  CHKERR(glDeleteBuffers(1, &mFieldModelConstantsBuffer));
  CHKERR(glDeleteBuffers(1, &mSolenoidSegmentsBuffer));
  CHKERR(glDeleteBuffers(1, &mSolenoidParameterizationBuffer));
  CHKERR(glDeleteBuffers(1, &mDipoleSegmentsBuffer));
  CHKERR(glDeleteBuffers(1, &mDipoleParameterizationBuffer));
  CHKERR(glDeleteProgram(mShaderProgramField));
  CHKERR(glDeleteShader(mGeometryShader));
  CHKERR(glDeleteShader(mVertexShaderPassthrough));
  CHKERR(glDeleteBuffers(1, &VBO_field));
  CHKERR(glDeleteVertexArrays(1, &VAO_field));
}

#else  // GPUCA_BUILD_EVENT_DISPLAY_OPENGL
int32_t GPUDisplayBackendOpenGL::InitMagFieldVisualization()
{
  return 0;
}
uint32_t GPUDisplayBackendOpenGL::drawField() { return 0; }
void GPUDisplayBackendOpenGL::ExitMagFieldVisualization() {}
#endif // GPUCA_BUILD_EVENT_DISPLAY_OPENGL
