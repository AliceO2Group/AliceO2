// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "GL/gl3w.h"
#include "GL/glcorearb.h"

#include "HandMadeMath.h"
#include <GLFW/glfw3.h>

#include <exception>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

namespace o2
{
namespace framework
{
namespace gl
{

GLuint compileShaders(const char* VertexSourcePointer, const char* FragmentSourcePointer)
{
  // Create the shaders
  GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
  GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

  GLint Result = GL_FALSE;
  int InfoLogLength;

  // Compile Vertex Shader
  glShaderSource(VertexShaderID, 1, &VertexSourcePointer, nullptr);
  glCompileShader(VertexShaderID);

  // Check Vertex Shader
  glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
  glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);

  if (InfoLogLength > 0) {
    std::vector<char> VertexShaderErrorMessage(InfoLogLength + 1);
    glGetShaderInfoLog(VertexShaderID, InfoLogLength, nullptr, &VertexShaderErrorMessage[0]);
    throw std::runtime_error(&VertexShaderErrorMessage[0]);
  }

  // Compile Fragment Shader
  glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer, nullptr);
  glCompileShader(FragmentShaderID);

  // Check Fragment Shader
  glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
  glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
  if (InfoLogLength > 0) {
    std::vector<char> FragmentShaderErrorMessage(InfoLogLength + 1);
    glGetShaderInfoLog(FragmentShaderID, InfoLogLength, nullptr, &FragmentShaderErrorMessage[0]);
    throw std::runtime_error(&FragmentShaderErrorMessage[0]);
  }

  // Link the program
  GLuint ProgramID = glCreateProgram();
  glAttachShader(ProgramID, VertexShaderID);
  glAttachShader(ProgramID, FragmentShaderID);
  glLinkProgram(ProgramID);

  // Check the program
  glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
  glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
  if (InfoLogLength > 0) {
    std::vector<char> ProgramErrorMessage(InfoLogLength + 1);
    glGetProgramInfoLog(ProgramID, InfoLogLength, nullptr, &ProgramErrorMessage[0]);
    throw std::runtime_error(&ProgramErrorMessage[0]);
  }

  glDetachShader(ProgramID, VertexShaderID);
  glDetachShader(ProgramID, FragmentShaderID);

  glDeleteShader(VertexShaderID);
  glDeleteShader(FragmentShaderID);

  return ProgramID;
}

static char const* fragmentShader = R"shader(#version 330 core
out vec3 color;
void main()
{
  vec2 coord = gl_PointCoord - vec2(0.5);  //from [0,1] to [-0.5,0.5]
  if(length(coord) > 0.5)                  //outside of circle radius?
      discard;

  color = vec3(1, 1, 1);
}
)shader";

static char const* vertexShader = R"shader(#version 330 core
layout(location = 0) in vec3 vertexPosition_modelspace;

uniform mat4 MVP;

void main() {
  gl_PointSize = 10.0;
  gl_Position =  MVP * vec4(vertexPosition_modelspace,1);
}
)shader";

static unsigned int sProgramId;
static unsigned int sMVPId;

void init3DContext(void* context)
{
  // Simply to show what the context is.
  GLFWwindow* w = (GLFWwindow*)(context);
  (void)w;
  sProgramId = compileShaders(vertexShader, fragmentShader);
  sMVPId = glGetUniformLocation(sProgramId, "MVP");
}

void render3D()
{
  unsigned int vaoID;
  unsigned int vboID;
  float vert[9];

  vert[0] = 0.0;
  vert[1] = 0.5;
  vert[2] = -1.0;
  vert[3] = -1.0;
  vert[4] = -0.5;
  vert[5] = -1.0;
  vert[6] = 1.0;
  vert[7] = -0.5;
  vert[8] = -1.0;

  glGenVertexArrays(1, &vaoID);
  glBindVertexArray(vaoID);

  glGenBuffers(1, &vboID);
  glBindBuffer(GL_ARRAY_BUFFER, vboID);

  glBufferData(GL_ARRAY_BUFFER, 9 * sizeof(GLfloat), vert, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(0);
  glEnable(GL_PROGRAM_POINT_SIZE);

  glUseProgram(sProgramId);

  int w = 1;
  int h = 1;

  hmm_mat4 proj = HMM_Perspective(60.0f, w / h, 0.01f, 10.0f);
  hmm_mat4 view = HMM_LookAt(HMM_Vec3(0.0f, 1.5f, 6.0f), HMM_Vec3(0.0f, 0.0f, 0.0f), HMM_Vec3(0.0f, -1.0f, 0.0f));
  hmm_mat4 view_proj = HMM_MultiplyMat4(proj, view);
  glUniformMatrix4fv(sMVPId, 1, GL_FALSE, &view_proj.Elements[0][0]);

  glDrawArrays(GL_POINTS, 0, 3);

  glDisableVertexAttribArray(0);
}

} // namespace gl
} // namespace framework
} // namespace o2
