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

/// \file GPUDisplayBackendVulkan.h
/// \author David Rohr

#ifndef GPUDISPLAYBACKENDVULKAN_H
#define GPUDISPLAYBACKENDVULKAN_H

#include "GPUDisplayBackend.h"
#include <vulkan/vulkan.hpp>

#include <vector>

namespace GPUCA_NAMESPACE
{
namespace gpu
{

struct QueueFamiyIndices;
struct SwapChainSupportDetails;
struct VulkanBuffer;

class GPUDisplayBackendVulkan : public GPUDisplayBackend
{
 public:
  GPUDisplayBackendVulkan();
  ~GPUDisplayBackendVulkan();

  int ExtInit() override;
  bool CoreProfile() override;
  unsigned int DepthBits() override;

 protected:
  void createFB(GLfb& fb, bool tex, bool withDepth, bool msaa) override;
  void deleteFB(GLfb& fb) override;

  unsigned int drawVertices(const vboList& v, const drawType t) override;
  void ActivateColor(std::array<float, 3>& color) override;
  void setQuality() override;
  void setDepthBuffer() override;
  void setFrameBuffer(int updateCurrent, unsigned int newID) override;
  int InitBackend() override;
  void ExitBackend() override;
  void clearScreen(bool colorOnly = false) override;
  void updateSettings() override;
  void loadDataToGPU(size_t totalVertizes) override;
  void prepareDraw() override;
  void finishDraw() override;
  void prepareText() override;
  void setMatrices(const hmm_mat4& proj, const hmm_mat4& view) override;
  void mixImages(GLfb& mixBuffer, float mixSlaveImage) override;
  void renderOffscreenBuffer(GLfb& buffer, GLfb& bufferNoMSAA, int mainBuffer) override;
  void readPixels(unsigned char* pixels, bool needBuffer, unsigned int width, unsigned int height) override;
  void pointSizeFactor(float factor) override;
  void lineWidthFactor(float factor) override;
  backendTypes backendType() const override { return TYPE_VULKAN; }
  void resizeScene(unsigned int width, unsigned int height) override;

  VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
  VulkanBuffer createBuffer(size_t size, const void* srcData = nullptr, VkBufferUsageFlags type = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, int deviceMemory = 1);
  void writeToBuffer(VulkanBuffer& buffer, size_t size, const void* srcData);
  void clearBuffer(VulkanBuffer& buffer);
  void clearVertexBuffers();
  void startFillCommandBuffer(VkCommandBuffer& commandBuffer, unsigned int imageIndex);
  void endFillCommandBuffer(VkCommandBuffer& commandBuffer, unsigned int imageIndex);
  void updateSwapChainDetails(const VkPhysicalDevice& device);
  void createDevice();
  void createPipeline();
  void createSwapChain();
  void createShaders();
  void createUniformLayouts();
  void clearDevice();
  void clearPipeline();
  void clearSwapChain();
  void clearShaders();
  void clearUniformLayouts();
  void recreateSwapChain();
  void needRecordCommandBuffers();

  unsigned int mIndirectId;
  std::vector<unsigned int> mVBOId;
  std::unique_ptr<QueueFamiyIndices> mQueueFamilyIndices;
  std::unique_ptr<SwapChainSupportDetails> mSwapChainDetails;
  int mModelViewProjId;
  int mColorId;

  bool mEnableValidationLayers = false;
  VkInstance mInstance;
  VkDebugUtilsMessengerEXT mDebugMessenger;
  VkPhysicalDevice mPhysicalDevice;
  VkDevice mDevice;
  VkQueue mGraphicsQueue;
  VkSurfaceKHR mSurface;
  VkSurfaceFormatKHR mSurfaceFormat;
  VkPresentModeKHR mPresentMode;
  VkExtent2D mExtent;
  VkSwapchainKHR mSwapChain;
  std::vector<VkImage> mImages;
  std::vector<VkImageView> mImageViews;
  VkShaderModule mModuleVertex;
  VkShaderModule mModuleFragment;
  VkPipelineLayout mPipelineLayout;
  VkRenderPass mRenderPass;
  std::vector<VkPipeline> mPipelines;
  std::vector<VkFramebuffer> mFramebuffers;
  VkCommandPool mCommandPool;
  unsigned int mImageCount = 0;
  unsigned int mFramesInFlight = 0;
  int mCurrentFrame = 0;
  uint32_t mImageIndex;
  std::vector<VkCommandBuffer> mCommandBuffers;
  std::vector<bool> mCommandBufferUpToDate;
  std::vector<VkSemaphore> mImageAvailableSemaphore;
  std::vector<VkSemaphore> mRenderFinishedSemaphore;
  std::vector<VkFence> mInFlightFence;
  std::vector<VulkanBuffer> mUniformBuffersMat;
  std::vector<VulkanBuffer> mUniformBuffersCol;
  std::vector<VkDescriptorSet> mDescriptorSets;
  VkDescriptorSetLayout mUniformDescriptor;
  VkDescriptorPool mDescriptorPool;

  std::vector<VulkanBuffer> mVBO;
  unsigned int mNVBOCreated = 0;
  std::vector<VulkanBuffer> mIndirectCommandBuffer;
  bool mIndirectCommandBufferCreated = false;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
