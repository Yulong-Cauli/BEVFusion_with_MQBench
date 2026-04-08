// BEVPoolV2 TensorRT Plugin Implementation
// Matching BEVFusion's bev_pool interface

#include "bev_pool_v2_plugin.h"
#include <cstring>
#include <iostream>

// Constructor from ONNX attributes
// B: batch size, D: depth (Z), H: height (Y), W: width (X)
BEVPoolV2Plugin::BEVPoolV2Plugin(int B, int D, int H, int W)
    : mB(B), mD(D), mH(H), mW(W) {}

// Constructor from serialized data
BEVPoolV2Plugin::BEVPoolV2Plugin(const void* data, size_t length) {
    const char* buf = static_cast<const char*>(data);
    memcpy(&mB, buf, sizeof(int)); buf += sizeof(int);
    memcpy(&mD, buf, sizeof(int)); buf += sizeof(int);
    memcpy(&mH, buf, sizeof(int)); buf += sizeof(int);
    memcpy(&mW, buf, sizeof(int));
}

BEVPoolV2Plugin::~BEVPoolV2Plugin() {}

// Output dimensions: [B, D, H, W, C] where C comes from input x
nvinfer1::DimsExprs BEVPoolV2Plugin::getOutputDimensions(
    int32_t outputIndex,
    const nvinfer1::DimsExprs* inputs,
    int32_t nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
    
    // inputs[0] = x: [N, C] - flattened features
    // inputs[1] = geom_feats: [N, 4] - coordinates
    // inputs[2] = interval_starts: [M]
    // inputs[3] = interval_lengths: [M]
    
    nvinfer1::DimsExprs output;
    output.nbDims = 5;
    output.d[0] = exprBuilder.constant(mB);  // B
    output.d[1] = exprBuilder.constant(mD);  // D (Z)
    output.d[2] = exprBuilder.constant(mH);  // H (Y)
    output.d[3] = exprBuilder.constant(mW);  // W (X)
    output.d[4] = inputs[0].d[1];            // C from x
    
    return output;
}

bool BEVPoolV2Plugin::supportsFormatCombination(
    int32_t pos,
    const nvinfer1::PluginTensorDesc* inOut,
    int32_t nbInputs,
    int32_t nbOutputs) noexcept {
    
    // Support FP32 for all inputs/outputs
    // pos 0: x [N, C], pos 1: geom_feats [N, 4], pos 2: interval_starts [M], pos 3: interval_lengths [M]
    // pos 4: output [B, D, H, W, C]
    if (pos < nbInputs) {
        // Inputs: x is FP32, others are INT32
        if (pos == 0) {
            return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
                   inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
        } else {
            return inOut[pos].type == nvinfer1::DataType::kINT32 &&
                   inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
        }
    } else {
        // Output
        return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
               inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    }
}

void BEVPoolV2Plugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int32_t nbOutputs) noexcept {
    // No dynamic configuration needed
}

size_t BEVPoolV2Plugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs,
    int32_t nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int32_t nbOutputs) const noexcept {
    return 0;  // No workspace needed
}

int32_t BEVPoolV2Plugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
    
    // Get dimensions
    const auto& x_dims = inputDesc[0].dims;           // [N, C]
    const auto& interval_dims = inputDesc[3].dims;    // [M]
    
    int N = x_dims.d[0];
    int C = x_dims.d[1];
    int n_intervals = interval_dims.d[0];
    
    const float* x = static_cast<const float*>(inputs[0]);
    const int* geom_feats = static_cast<const int*>(inputs[1]);
    const int* interval_starts = static_cast<const int*>(inputs[2]);
    const int* interval_lengths = static_cast<const int*>(inputs[3]);
    float* out = static_cast<float*>(outputs[0]);
    
    launch_bev_pool_v2(
        mB, mD, mH, mW, n_intervals, C,
        x, geom_feats, interval_starts, interval_lengths,
        out, stream
    );
    
    return 0;
}

nvinfer1::DataType BEVPoolV2Plugin::getOutputDataType(
    int32_t index,
    const nvinfer1::DataType* inputTypes,
    int32_t nbInputs) const noexcept {
    return nvinfer1::DataType::kFLOAT;
}

const char* BEVPoolV2Plugin::getPluginType() const noexcept {
    return BEV_POOL_PLUGIN_NAME;
}

const char* BEVPoolV2Plugin::getPluginVersion() const noexcept {
    return BEV_POOL_PLUGIN_VERSION;
}

int32_t BEVPoolV2Plugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t BEVPoolV2Plugin::initialize() noexcept {
    return 0;
}

void BEVPoolV2Plugin::terminate() noexcept {}

size_t BEVPoolV2Plugin::getSerializationSize() const noexcept {
    return 4 * sizeof(int);  // B, D, H, W
}

void BEVPoolV2Plugin::serialize(void* buffer) const noexcept {
    char* buf = static_cast<char*>(buffer);
    memcpy(buf, &mB, sizeof(int)); buf += sizeof(int);
    memcpy(buf, &mD, sizeof(int)); buf += sizeof(int);
    memcpy(buf, &mH, sizeof(int)); buf += sizeof(int);
    memcpy(buf, &mW, sizeof(int));
}

void BEVPoolV2Plugin::destroy() noexcept {
    delete this;
}

void BEVPoolV2Plugin::setPluginNamespace(const char* pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}

const char* BEVPoolV2Plugin::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

nvinfer1::IPluginV2DynamicExt* BEVPoolV2Plugin::clone() const noexcept {
    return new BEVPoolV2Plugin(mB, mD, mH, mW);
}

// ==================== Plugin Creator ====================

BEVPoolV2PluginCreator::BEVPoolV2PluginCreator() {
    mFields = {
        {"B", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
        {"D", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
        {"H", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
        {"W", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
    };
    mFC.nbFields = static_cast<int32_t>(mFields.size());
    mFC.fields = mFields.data();
}

const char* BEVPoolV2PluginCreator::getPluginName() const noexcept {
    return BEV_POOL_PLUGIN_NAME;
}

const char* BEVPoolV2PluginCreator::getPluginVersion() const noexcept {
    return BEV_POOL_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* BEVPoolV2PluginCreator::getFieldNames() noexcept {
    return &mFC;
}

nvinfer1::IPluginV2* BEVPoolV2PluginCreator::createPlugin(
    const char* name,
    const nvinfer1::PluginFieldCollection* fc) noexcept {
    
    int B = 1, D = 118, H = 128, W = 128;  // Default values
    
    for (int i = 0; i < fc->nbFields; ++i) {
        const auto& f = fc->fields[i];
        if (!f.data) continue;  // Skip null data
        
        if (std::string(f.name) == "B") {
            B = *static_cast<const int*>(f.data);
        } else if (std::string(f.name) == "D") {
            D = *static_cast<const int*>(f.data);
        } else if (std::string(f.name) == "H") {
            H = *static_cast<const int*>(f.data);
        } else if (std::string(f.name) == "W") {
            W = *static_cast<const int*>(f.data);
        }
    }
    
    return new BEVPoolV2Plugin(B, D, H, W);
}

nvinfer1::IPluginV2* BEVPoolV2PluginCreator::deserializePlugin(
    const char* name,
    const void* serialData,
    size_t serialLength) noexcept {
    return new BEVPoolV2Plugin(serialData, serialLength);
}

void BEVPoolV2PluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}

const char* BEVPoolV2PluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

// Explicit initialization function for loading .so
extern "C" {
    __attribute__((visibility("default")))
    void forceInitBEVPoolV2Plugin() {
        // Empty: actual registration done by REGISTER_TENSORRT_PLUGIN
    }
}

// Auto-registration
REGISTER_TENSORRT_PLUGIN(BEVPoolV2PluginCreator);
