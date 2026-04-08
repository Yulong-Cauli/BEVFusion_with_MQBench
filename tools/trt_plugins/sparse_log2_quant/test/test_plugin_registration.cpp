/**
 * @file test_plugin_registration.cpp
 * @brief 简单的 C++ 测试程序，验证 SparseLog2Quant Plugin 在 TRT 8.6 SDK 下能正确注册。
 *
 * 编译命令：
 *   g++ -o test_plugin_registration test_plugin_registration.cpp \
 *       -I/media/yellowstone/databig2/gzj/tensorrt/TensorRT-8.6.1.6/include \
 *       -L/media/yellowstone/databig2/gzj/tensorrt/TensorRT-8.6.1.6/lib \
 *       -lnvinfer -lcudart \
 *       -L../build -lsparse_log2_quant_plugin \
 *       -Wl,-rpath,../build
 *
 * 运行命令：
 *   ./test_plugin_registration
 *
 * 期望输出：
 *   [OK] Plugin registry initialized
 *   [OK] Plugin 'SparseLog2Quant' version '1' found in registry
 *   [OK] Plugin creator retrieved successfully
 *   [OK] All tests passed!
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "NvInfer.h"
#include "NvInferPlugin.h"

// 显式初始化函数声明（来自 Plugin .so）
extern "C" void forceInitSparseLog2QuantPlugin();

int main(int argc, char** argv) {
    printf("=== SparseLog2Quant Plugin Registration Test ===\n\n");

    // ── Step 1: 初始化 TRT Logger ─────────────────────────────────────────────
    class TestLogger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING) {
                printf("[TRT] %s\n", msg);
            }
        }
    };
    static TestLogger gLogger;

    // ── Step 2: 确保 Plugin .so 已加载 ──────────────────────────────────────
    printf("Step 1: Loading plugin library...\n");
    // Plugin 注册由 REGISTER_TENSORRT_PLUGIN 宏在 .so 加载时自动完成
    // forceInitSparseLog2QuantPlugin() 仅用于强制加载 .so，不执行注册
    // 在本测试中，通过链接器链接了 libsparse_log2_quant_plugin.so，
    // 全局构造函数会自动执行注册
    forceInitSparseLog2QuantPlugin();
    printf("[OK] Plugin library loaded (registration done by REGISTER_TENSORRT_PLUGIN)\n\n");

    // ── Step 3: 获取 Plugin Registry ─────────────────────────────────────────
    printf("Step 2: Getting plugin registry...\n");
    auto* registry = getPluginRegistry();
    if (registry == nullptr) {
        printf("[FAIL] Failed to get plugin registry\n");
        return 1;
    }
    printf("[OK] Plugin registry initialized\n\n");

    // ── Step 4: 查找 SparseLog2Quant Plugin ─────────────────────────────────
    printf("Step 3: Looking for 'SparseLog2Quant' plugin...\n");
    auto* creator = registry->getPluginCreator("SparseLog2Quant", "1");
    if (creator == nullptr) {
        printf("[FAIL] Plugin 'SparseLog2Quant' version '1' NOT found in registry\n");
        printf("       Make sure libsparse_log2_quant_plugin.so is properly linked/loaded.\n");
        return 1;
    }
    printf("[OK] Plugin 'SparseLog2Quant' version '1' found in registry\n");
    printf("     Plugin name: %s\n", creator->getPluginName());
    printf("     Plugin version: %s\n", creator->getPluginVersion());
    printf("     Plugin namespace: %s\n\n", creator->getPluginNamespace());

    // ── Step 5: 获取 Plugin 字段信息 ─────────────────────────────────────────
    printf("Step 4: Checking plugin field names...\n");
    const auto* fields = creator->getFieldNames();
    if (fields == nullptr) {
        printf("[FAIL] Failed to get field names\n");
        return 1;
    }
    printf("[OK] Plugin supports %d field(s):\n", fields->nbFields);
    for (int i = 0; i < fields->nbFields; ++i) {
        printf("     - %s\n", fields->fields[i].name);
    }
    printf("\n");

    // ── Step 6: 尝试创建 Plugin 实例（使用测试参数）──────────────────────────
    printf("Step 5: Creating plugin instance with test parameters...\n");

    // 准备测试参数：log2_base = [-2.0], per_channel = false
    float log2_base_val = -2.0f;
    int32_t per_channel_val = 0;

    nvinfer1::PluginField fields_data[2] = {
        {"log2_base", &log2_base_val, nvinfer1::PluginFieldType::kFLOAT32, 1},
        {"per_channel", &per_channel_val, nvinfer1::PluginFieldType::kINT32, 1}
    };

    nvinfer1::PluginFieldCollection fc;
    fc.nbFields = 2;
    fc.fields = fields_data;

    auto* plugin = creator->createPlugin("test_plugin", &fc);
    if (plugin == nullptr) {
        printf("[FAIL] Failed to create plugin instance\n");
        return 1;
    }
    printf("[OK] Plugin instance created successfully\n");
    printf("     Plugin type: %s\n", plugin->getPluginType());
    printf("     Plugin version: %s\n", plugin->getPluginVersion());
    printf("     Number of outputs: %d\n\n", plugin->getNbOutputs());

    // ── Step 7: 测试序列化和反序列化 ─────────────────────────────────────────
    printf("Step 6: Testing serialization/deserialization...\n");
    size_t serial_size = plugin->getSerializationSize();
    printf("     Serialization size: %zu bytes\n", serial_size);

    void* serial_data = malloc(serial_size);
    if (serial_data == nullptr) {
        printf("[FAIL] Failed to allocate memory for serialization\n");
        plugin->destroy();
        return 1;
    }

    plugin->serialize(serial_data);
    printf("[OK] Plugin serialized successfully\n");

    // 使用反序列化创建新实例
    auto* plugin_deserialized = creator->deserializePlugin("test_plugin_deser", serial_data, serial_size);
    if (plugin_deserialized == nullptr) {
        printf("[FAIL] Failed to deserialize plugin\n");
        free(serial_data);
        plugin->destroy();
        return 1;
    }
    printf("[OK] Plugin deserialized successfully\n");
    printf("     Deserialized plugin type: %s\n\n", plugin_deserialized->getPluginType());

    // ── 清理 ─────────────────────────────────────────────────────────────────
    free(serial_data);
    plugin->destroy();
    plugin_deserialized->destroy();

    // ── 总结 ─────────────────────────────────────────────────────────────────
    printf("=================================================\n");
    printf("[SUCCESS] All tests passed!\n");
    printf("\nThe SparseLog2Quant Plugin is correctly registered\n");
    printf("and functional in C++ TRT 8.6 SDK environment.\n");
    printf("=================================================\n");

    return 0;
}
