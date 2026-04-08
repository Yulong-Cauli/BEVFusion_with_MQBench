/**
 * @file sparse_log2_quant_plugin.h
 * @brief TensorRT Plugin for SparseLog2Quant (对数域稀疏量化).
 *
 * 用途：lidar/backbone 的稀疏卷积特征量化。
 *       激活呈二模态分布（大多数为 0），对数域量化比线性 INT8 精度高很多。
 *
 * 数学实现：
 *   zero_mask = |x| < eps
 *   log2_x = log2(|x|) - base
 *   q = round(log2_x).clamp(-127, 127)
 *   x_dq = sign(x) * 2^(q + base)
 *   x_dq = where(zero_mask, 0, x_dq)
 *
 * 支持：
 *   - 输入形状：[N, C] 或 [N, C, H, W]（通过 IPluginV2DynamicExt 动态支持）
 *   - Per-tensor（log2_base 是 scalar）或 Per-channel（log2_base 是 [C] 数组）
 *   - 输入/输出数据类型：FP16
 */

#pragma once

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <string>
#include <vector>
#include <cuda_runtime.h>

// Plugin 名称和版本
static const char* LOG2_PLUGIN_NAME    = "SparseLog2Quant";
static const char* LOG2_PLUGIN_VERSION = "1";

// CUDA kernel 入口（在 sparse_log2_quant_kernel.cu 中实现）
extern "C" void launch_sparse_log2_quant(
    const void* input,
    void* output,
    const float* log2_base,
    int32_t dim0,             // N
    int32_t dim1,             // C
    int32_t dim2,             // H or D
    int32_t dim3,             // W
    int32_t nb_dims,          // 维度数量
    bool per_channel,
    cudaStream_t stream
);

/**
 * @brief SparseLog2Quant TRT Plugin 实现.
 *
 * 使用 IPluginV2DynamicExt 支持动态 shape。
 */
class SparseLog2QuantPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    /**
     * @brief 构造函数（从参数创建）.
     * @param log2_base 对数域 base，per-tensor 时长度为 1，per-channel 时长度为 C
     * @param per_channel 是否逐通道量化
     */
    SparseLog2QuantPlugin(std::vector<float> log2_base, bool per_channel);

    /**
     * @brief 构造函数（从序列化数据反序列化）.
     */
    SparseLog2QuantPlugin(const void* data, size_t length);

    ~SparseLog2QuantPlugin() override;

    // ── IPluginV2DynamicExt 接口 ─────────────────────────────────────────────

    /**
     * @brief 获取输出维度（与输入相同）.
     */
    nvinfer1::DimsExprs getOutputDimensions(
        int32_t outputIndex,
        const nvinfer1::DimsExprs* inputs,
        int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder
    ) noexcept override;

    /**
     * @brief 检查格式组合支持.
     * 仅支持 FP16 + kLINEAR 格式。
     */
    bool supportsFormatCombination(
        int32_t pos,
        const nvinfer1::PluginTensorDesc* inOut,
        int32_t nbInputs,
        int32_t nbOutputs
    ) noexcept override;

    /**
     * @brief 配置 Plugin（分配 GPU 内存）.
     */
    void configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* in,
        int32_t nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out,
        int32_t nbOutputs
    ) noexcept override;

    /**
     * @brief 获取工作空间大小（本 Plugin 不需要额外工作空间）.
     */
    size_t getWorkspaceSize(
        const nvinfer1::PluginTensorDesc* inputs,
        int32_t nbInputs,
        const nvinfer1::PluginTensorDesc* outputs,
        int32_t nbOutputs
    ) const noexcept override;

    /**
     * @brief 执行量化（核心逻辑）.
     */
    int32_t enqueue(
        const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream
    ) noexcept override;

    /**
     * @brief 获取输出数据类型（与输入相同）.
     */
    nvinfer1::DataType getOutputDataType(
        int32_t index,
        const nvinfer1::DataType* inputTypes,
        int32_t nbInputs
    ) const noexcept override;

    // ── IPluginV2 接口 ─────────────────────────────────────────────────────

    const char* getPluginType()    const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int32_t     getNbOutputs()     const noexcept override;
    int32_t     initialize()              noexcept override;
    void        terminate()               noexcept override;
    size_t      getSerializationSize()    const noexcept override;
    void        serialize(void* buffer)   const noexcept override;
    void        destroy()                       noexcept override;
    void        setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace()       const noexcept override;
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

private:
    std::vector<float> mLog2Base;       // 对数域 base [1] 或 [C]
    bool               mPerChannel;     // 是否逐通道
    float*             mLog2BaseDevice; // GPU 上的 log2_base
    std::string        mNamespace;      // Plugin namespace
    int32_t            mCachedC;        // 缓存的通道数（用于动态 shape）
};

/**
 * @brief Plugin Creator，用于从 ONNX 节点创建 Plugin 实例.
 */
class SparseLog2QuantPluginCreator : public nvinfer1::IPluginCreator {
public:
    SparseLog2QuantPluginCreator();

    const char* getPluginName()    const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(
        const char* name,
        const nvinfer1::PluginFieldCollection* fc
    ) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(
        const char* name,
        const void* serialData,
        size_t serialLength
    ) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    nvinfer1::PluginFieldCollection    mFC;
    std::vector<nvinfer1::PluginField> mFields;
    std::string                        mNamespace;
};
