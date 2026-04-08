// BEVPoolV2 TensorRT Plugin
// Adapted for BEVFusion's existing interval-sum kernel
//
// Input format:
//   - x: [N, C] flattened features after outer product
//   - geom_feats: [N, 4] voxel coordinates (x, y, z, batch)
//   - interval_starts: [M] interval start indices
//   - interval_lengths: [M] interval lengths
//
// Output: [B, D, H, W, C] BEV features (where D=Z, H=Y, W=X)

#pragma once
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <string>
#include <vector>
#include <cuda_runtime.h>

static const char* BEV_POOL_PLUGIN_NAME = "BEVPoolV2";
static const char* BEV_POOL_PLUGIN_VERSION = "1";

// CUDA kernel interface matching BEVFusion's kernel
// Args:
//   b, d, h, w: output dimensions (batch, depth/Z, height/Y, width/X)
//   n_intervals: number of intervals
//   c: number of channels
//   x: [N, C] input features
//   geom_feats: [N, 4] coordinates
//   interval_starts: [M] interval start indices
//   interval_lengths: [M] interval lengths
//   out: [B, D, H, W, C] output
//   stream: CUDA stream
extern "C" void launch_bev_pool_v2(
    int b, int d, int h, int w, int n_intervals, int c,
    const float* x,
    const int* geom_feats,
    const int* interval_starts,
    const int* interval_lengths,
    float* out, cudaStream_t stream);

class BEVPoolV2Plugin : public nvinfer1::IPluginV2DynamicExt {
public:
    // Constructor for creating from ONNX
    // B, D, H, W: output dimensions (batch, depth/Z, height/Y, width/X)
    BEVPoolV2Plugin(int B, int D, int H, int W);
    
    // Constructor for deserialization
    BEVPoolV2Plugin(const void* data, size_t length);
    
    ~BEVPoolV2Plugin() override;

    // IPluginV2DynamicExt methods
    nvinfer1::DimsExprs getOutputDimensions(
        int32_t outputIndex,
        const nvinfer1::DimsExprs* inputs,
        int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(
        int32_t pos,
        const nvinfer1::PluginTensorDesc* inOut,
        int32_t nbInputs,
        int32_t nbOutputs) noexcept override;

    void configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* in,
        int32_t nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out,
        int32_t nbOutputs) noexcept override;

    size_t getWorkspaceSize(
        const nvinfer1::PluginTensorDesc* inputs,
        int32_t nbInputs,
        const nvinfer1::PluginTensorDesc* outputs,
        int32_t nbOutputs) const noexcept override;

    int32_t enqueue(
        const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream) noexcept override;

    nvinfer1::DataType getOutputDataType(
        int32_t index,
        const nvinfer1::DataType* inputTypes,
        int32_t nbInputs) const noexcept override;

    // IPluginV2Ext methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

private:
    // Output dimensions: [B, D, H, W, C] where D=Z, H=Y, W=X
    int mB, mD, mH, mW;
    std::string mNamespace;
};

class BEVPoolV2PluginCreator : public nvinfer1::IPluginCreator {
public:
    BEVPoolV2PluginCreator();
    
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(
        const char* name,
        const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(
        const char* name,
        const void* serialData,
        size_t serialLength) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mFields;
    std::string mNamespace;
};
