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
#define GLFW_INCLUDE_NONE

#define SOKOL_IMPL
#define SOKOL_GLCORE33
#include "sokol_gfx.h"

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
namespace sokol
{

static char const* fragmentShader = R"shader(#version 330 core
in vec4 color;
out vec4 frag_color;

void main()
{
  frag_color = color;
}
)shader";

static char const* vertexShader = R"shader(#version 330 core
in vec3 position;
in vec4 color0;
out vec4 color;

uniform mat4 MVP;

void main() {
  gl_Position = MVP * vec4(position,1);
  color = color0;
}
)shader";

static unsigned int sProgramId;
static unsigned int sMVPId;

struct vs_params_t {
  hmm_mat4 mvp;
};

struct DrawCmd {
  size_t drawStateIdx;
  size_t uniformIdx;

  int start;
  int end;
  int instances;
};

struct Context3D {
  GLFWwindow* window;
  std::vector<sg_buffer> buffers;
  std::vector<sg_shader> shaders;
  std::vector<sg_pipeline> pipelines;
  std::vector<sg_draw_state> drawStates;
  std::vector<sg_pass_action> passes;
  std::vector<vs_params_t> params;
  std::vector<DrawCmd> commands;
};

static Context3D g3dCtx;

void init3DContext(void* context)
{
  GLFWwindow* w = (GLFWwindow*)(context);
  int cur_width, cur_height;
  glfwGetFramebufferSize(w, &cur_width, &cur_height);
  g3dCtx.window = w;

  sg_desc description{0};
  sg_setup(&description);

  const float vertices[] = {
    // positions            // colors
    0.0f, 0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
    0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f,
    -0.5f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f};

  sg_buffer_desc vbuf_desc{0};
  vbuf_desc.size = sizeof(vertices);
  vbuf_desc.content = vertices;
  g3dCtx.buffers.push_back(sg_make_buffer(&vbuf_desc));

  sg_shader_desc points_shader{0};
  points_shader.vs.source = vertexShader;
  points_shader.vs.uniform_blocks[0].size = sizeof(vs_params_t);
  points_shader.vs.uniform_blocks[0].uniforms[0].name = "MVP";
  points_shader.vs.uniform_blocks[0].uniforms[0].type = SG_UNIFORMTYPE_MAT4;
  points_shader.fs.source = fragmentShader;
  g3dCtx.shaders.push_back(sg_make_shader(&points_shader));

  sg_pipeline_desc points_pipeline{0};
  points_pipeline.shader = g3dCtx.shaders.back();
  points_pipeline.layout.attrs[0].name = "position";
  points_pipeline.layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT3;
  points_pipeline.layout.attrs[1].name = "color0";
  points_pipeline.layout.attrs[1].format = SG_VERTEXFORMAT_FLOAT3;

  g3dCtx.pipelines.push_back(sg_make_pipeline(&points_pipeline));

  sg_draw_state state{0};
  state.pipeline = g3dCtx.pipelines.back();
  state.vertex_buffers[0] = g3dCtx.buffers.back();

  g3dCtx.drawStates.push_back(state);

  hmm_mat4 proj = HMM_Perspective(60.0f, cur_width / cur_height, 0.01f, 10.0f);
  hmm_mat4 view = HMM_LookAt(HMM_Vec3(0.0f, 1.5f, 6.0f), HMM_Vec3(0.0f, 0.0f, 0.0f), HMM_Vec3(0.0f, -1.0f, 0.0f));
  hmm_mat4 view_proj = HMM_MultiplyMat4(proj, view);
  g3dCtx.params.push_back(vs_params_t{view_proj});

  // Cleaning happens somewhere else
  sg_pass_action default_pass_action{0};
  default_pass_action.colors[0].action = SG_ACTION_DONTCARE;
  default_pass_action.colors[1].action = SG_ACTION_DONTCARE;
  default_pass_action.colors[1].action = SG_ACTION_DONTCARE;
  default_pass_action.depth.action = SG_ACTION_DONTCARE;
  default_pass_action.stencil.action = SG_ACTION_DONTCARE;
  g3dCtx.passes.push_back(default_pass_action);

  g3dCtx.commands.push_back({g3dCtx.drawStates.size() - 1, g3dCtx.params.size() - 1, 0, 3, 1});
}

void render3D()
{
  int cur_width, cur_height;
  glfwGetFramebufferSize(g3dCtx.window, &cur_width, &cur_height);

  int lastDrawState = -1;
  int lastUniform = -1;

  // Cleaning happens somewhere else
  sg_begin_default_pass(&g3dCtx.passes.back(), cur_width, cur_height);

  for (auto& command : g3dCtx.commands) {
    /// We switch draw state only when needed.
    if (command.drawStateIdx != lastDrawState) {
      sg_draw_state const& state = g3dCtx.drawStates[command.drawStateIdx];
      lastDrawState = command.drawStateIdx;
      sg_apply_draw_state(&state);
    }
    /// We switch uniforms only when needed.
    if (command.uniformIdx != lastUniform) {
      vs_params_t const& params = g3dCtx.params[command.uniformIdx];
      sg_apply_uniform_block(SG_SHADERSTAGE_VS, 0, &params, sizeof(params));
      lastUniform = command.uniformIdx;
    }
    /// Do the drawing
    sg_draw(command.start, command.end, command.instances);
  }
  sg_end_pass();
  sg_commit();
}

} // namespace sokol
} // namespace framework
} // namespace o2
