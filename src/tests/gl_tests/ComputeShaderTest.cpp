//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ComputeShaderTest:
//   Compute shader specific tests.
//

#include <array>
#include <sstream>
#include <string>
#include <vector>
#include "common/debug.h"
#include "common/span.h"
#include "test_utils/ANGLETest.h"
#include "test_utils/gl_raii.h"

using namespace angle;

namespace
{

class ComputeShaderTest : public ANGLETest<>
{
  protected:
    ComputeShaderTest() {}

    void createMockOutputImage(GLuint texture, GLenum internalFormat, GLint width, GLint height)
    {
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexStorage2D(GL_TEXTURE_2D, 1, internalFormat, width, height);
        EXPECT_GL_NO_ERROR();

        glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, internalFormat);
        EXPECT_GL_NO_ERROR();
    }

    template <class T, GLint kWidth, GLint kHeight>
    void runSharedMemoryTest(const char *kCS,
                             GLenum internalFormat,
                             GLenum format,
                             const std::array<T, kWidth * kHeight> &inputData,
                             const std::array<T, kWidth * kHeight> &expectedValues)
    {
        GLTexture texture[2];
        GLFramebuffer framebuffer;

        glBindTexture(GL_TEXTURE_2D, texture[0]);
        glTexStorage2D(GL_TEXTURE_2D, 1, internalFormat, kWidth, kHeight);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kWidth, kHeight, GL_RED_INTEGER, format,
                        inputData.data());
        EXPECT_GL_NO_ERROR();

        constexpr T initData[kWidth * kHeight] = {};
        glBindTexture(GL_TEXTURE_2D, texture[1]);
        glTexStorage2D(GL_TEXTURE_2D, 1, internalFormat, kWidth, kHeight);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kWidth, kHeight, GL_RED_INTEGER, format, initData);
        EXPECT_GL_NO_ERROR();

        ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
        glUseProgram(program);

        glBindImageTexture(0, texture[0], 0, GL_FALSE, 0, GL_READ_ONLY, internalFormat);
        EXPECT_GL_NO_ERROR();

        glBindImageTexture(1, texture[1], 0, GL_FALSE, 0, GL_WRITE_ONLY, internalFormat);
        EXPECT_GL_NO_ERROR();

        glDispatchCompute(1, 1, 1);
        EXPECT_GL_NO_ERROR();

        glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);

        std::array<T, kWidth * kHeight> outputValues = {};
        glUseProgram(0);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);

        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture[1],
                               0);
        EXPECT_GL_NO_ERROR();
        glReadPixels(0, 0, kWidth, kHeight, GL_RED_INTEGER, format, outputValues.data());
        EXPECT_GL_NO_ERROR();

        for (int i = 0; i < kWidth * kHeight; i++)
        {
            EXPECT_EQ(expectedValues[i], outputValues[i]);
        }
    }
};

class ComputeShaderTestES3 : public ANGLETest<>
{
  protected:
    ComputeShaderTestES3() {}
};

class ExperimentalGles31MetalComputeTest : public ANGLETest<>
{};

PlatformParameters ExperimentalGles31Metal()
{
    PlatformParameters params(3, 1, egl_platform::METAL());
    params.enable(Feature::EnableExperimentalGles31);
    return params;
}

TEST_P(ExperimentalGles31MetalComputeTest, ComputeWorkGroupLimits)
{
    GLint maxInvocations = 0;
    glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &maxInvocations);
    EXPECT_GL_NO_ERROR();
    EXPECT_GE(maxInvocations, 128);
    EXPECT_LE(maxInvocations, 1024);
}

TEST_P(ExperimentalGles31MetalComputeTest, DispatchWithUniformsRuntimeArrayAndBarriers)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x = 4) in;
layout(std430, binding = 0) buffer Data
{
    uint baseValue;
    uint values[];
} data;
uniform uint increment;
uniform int initializeValues;
void main()
{
    uint i = gl_GlobalInvocationID.x;
    if (i < uint(data.values.length()))
    {
        if (initializeValues != 0)
            data.values[i] = data.baseValue + i * 7u + increment;
        else
            data.values[i] += increment;
    }
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);

    constexpr GLuint kElementCount = 16;
    GLBuffer buffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, (kElementCount + 1) * sizeof(GLuint), nullptr,
                 GL_DYNAMIC_DRAW);
    constexpr GLuint kBaseValue = 42;
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(kBaseValue), &kBaseValue);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer);
    glUseProgram(program);

    const GLint incrementLocation = glGetUniformLocation(program, "increment");
    const GLint initializeLocation = glGetUniformLocation(program, "initializeValues");
    ASSERT_NE(-1, incrementLocation);
    ASSERT_NE(-1, initializeLocation);

    glUniform1ui(incrementLocation, 23);
    glUniform1i(initializeLocation, 1);
    glDispatchCompute(4, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glUniform1ui(incrementLocation, 100);
    glUniform1i(initializeLocation, 0);
    glDispatchCompute(4, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    glFinish();
    const auto *values = static_cast<const GLuint *>(
        glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (kElementCount + 1) * sizeof(GLuint),
                         GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, values);
    for (GLuint i = 0; i < kElementCount; ++i)
    {
        EXPECT_EQ(kBaseValue + i * 7 + 123, values[i + 1]);
    }
    EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));
    EXPECT_GL_NO_ERROR();
}

TEST_P(ExperimentalGles31MetalComputeTest, IndirectDispatchWithGpuGeneratedArguments)
{
    constexpr char kGenerateArgsCS[] = R"(#version 310 es
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer DispatchArguments
{
    uint unused;
    uint groupsX;
    uint groupsY;
    uint groupsZ;
} args;
void main()
{
    args.groupsX = 5u;
    args.groupsY = 1u;
    args.groupsZ = 1u;
})";
    constexpr char kIndirectCS[]     = R"(#version 310 es
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Output
{
    uint values[];
} outputData;
void main()
{
    uint i = gl_GlobalInvocationID.x;
    outputData.values[i] = 0xCAFE0000u + i;
})";

    ANGLE_GL_COMPUTE_PROGRAM(generateArgsProgram, kGenerateArgsCS);
    ANGLE_GL_COMPUTE_PROGRAM(indirectProgram, kIndirectCS);

    GLBuffer argumentsBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, argumentsBuffer);
    const std::array<GLuint, 4> initialArguments = {};
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(initialArguments), initialArguments.data(),
                 GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, argumentsBuffer);
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, argumentsBuffer);

    glUseProgram(generateArgsProgram);
    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_COMMAND_BARRIER_BIT);

    constexpr GLuint kGroupCount = 5;
    GLBuffer outputBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, kGroupCount * sizeof(GLuint), nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, outputBuffer);

    glUseProgram(indirectProgram);
    glDispatchComputeIndirect(sizeof(GLuint));
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    glFinish();
    const auto *values = static_cast<const GLuint *>(glMapBufferRange(
        GL_SHADER_STORAGE_BUFFER, 0, kGroupCount * sizeof(GLuint), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, values);
    for (GLuint i = 0; i < kGroupCount; ++i)
    {
        EXPECT_EQ(0xCAFE0000u + i, values[i]);
    }
    EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));
    EXPECT_GL_NO_ERROR();
}

TEST_P(ExperimentalGles31MetalComputeTest, ComputeProgramPipeline)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Output
{
    uint values[];
} outputData;
uniform uint baseValue;
void main()
{
    uint i = gl_GlobalInvocationID.x;
    outputData.values[i] = baseValue + i;
})";

    GLProgram program;
    program.makeEmpty();
    GLShader shader(GL_COMPUTE_SHADER);
    const char *source = kCS;
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    ASSERT_GL_NO_ERROR();
    GLint compileStatus = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compileStatus);
    ASSERT_EQ(GL_TRUE, compileStatus);

    glProgramParameteri(program, GL_PROGRAM_SEPARABLE, GL_TRUE);
    glAttachShader(program, shader);
    glLinkProgram(program);
    GLint linkStatus = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
    ASSERT_EQ(GL_TRUE, linkStatus);
    ASSERT_GL_NO_ERROR();
    const GLint baseValueLocation = glGetUniformLocation(program, "baseValue");
    ASSERT_NE(-1, baseValueLocation);
    glProgramUniform1ui(program, baseValueLocation, 0xBEEF0000u);

    GLProgramPipeline pipeline;
    glUseProgramStages(pipeline, GL_COMPUTE_SHADER_BIT, program);
    glUseProgram(0);
    glBindProgramPipeline(pipeline);

    constexpr GLuint kElementCount = 8;
    GLBuffer buffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, kElementCount * sizeof(GLuint), nullptr,
                 GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer);

    glDispatchCompute(kElementCount, 1, 1);
    EXPECT_GL_NO_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    const auto *values = static_cast<const GLuint *>(glMapBufferRange(
        GL_SHADER_STORAGE_BUFFER, 0, kElementCount * sizeof(GLuint), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, values);
    for (GLuint i = 0; i < kElementCount; ++i)
    {
        EXPECT_EQ(0xBEEF0000u + i, values[i]);
    }
    EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));
    EXPECT_GL_NO_ERROR();
}

TEST_P(ExperimentalGles31MetalComputeTest, GraphicsProgramPipeline)
{
    constexpr char kVS[] = R"(#version 310 es
layout(location = 0) in vec2 position;
layout(location = 0) out vec4 color;
layout(std430, binding = 0) buffer VertexOutput
{
    uint values[];
} vertexOutput;
void main()
{
    vertexOutput.values[gl_VertexID] = uint(gl_VertexID) + 17u;
    color = vec4(1.0, 0.0, 0.0, 1.0);
    gl_Position = vec4(position, 0.0, 1.0);
})";
    constexpr char kFS[] = R"(#version 310 es
precision highp float;
layout(location = 0) in vec4 color;
layout(location = 0) out vec4 fragColor;
layout(std430, binding = 1) readonly buffer FragmentScale
{
    float value;
} fragmentScale;
uniform float intensity;
void main()
{
    fragColor = color * fragmentScale.value * intensity;
})";

    GLProgram vertexProgram;
    vertexProgram.makeEmpty();
    GLShader vertexShader(GL_VERTEX_SHADER);
    const char *vertexSource = kVS;
    glShaderSource(vertexShader, 1, &vertexSource, nullptr);
    glCompileShader(vertexShader);
    ASSERT_GL_NO_ERROR();
    GLint compileStatus = GL_FALSE;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &compileStatus);
    ASSERT_EQ(GL_TRUE, compileStatus);
    glProgramParameteri(vertexProgram, GL_PROGRAM_SEPARABLE, GL_TRUE);
    glAttachShader(vertexProgram, vertexShader);
    glLinkProgram(vertexProgram);
    GLint linkStatus = GL_FALSE;
    glGetProgramiv(vertexProgram, GL_LINK_STATUS, &linkStatus);
    ASSERT_EQ(GL_TRUE, linkStatus);

    GLProgram fragmentProgram;
    fragmentProgram.makeEmpty();
    GLShader fragmentShader(GL_FRAGMENT_SHADER);
    const char *fragmentSource = kFS;
    glShaderSource(fragmentShader, 1, &fragmentSource, nullptr);
    glCompileShader(fragmentShader);
    ASSERT_GL_NO_ERROR();
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &compileStatus);
    ASSERT_EQ(GL_TRUE, compileStatus);
    glProgramParameteri(fragmentProgram, GL_PROGRAM_SEPARABLE, GL_TRUE);
    glAttachShader(fragmentProgram, fragmentShader);
    glLinkProgram(fragmentProgram);
    glGetProgramiv(fragmentProgram, GL_LINK_STATUS, &linkStatus);
    ASSERT_EQ(GL_TRUE, linkStatus);
    ASSERT_GL_NO_ERROR();
    const GLint intensityLocation = glGetUniformLocation(fragmentProgram, "intensity");
    ASSERT_NE(-1, intensityLocation);
    glProgramUniform1f(fragmentProgram, intensityLocation, 1.0f);

    GLProgramPipeline pipeline;
    glUseProgramStages(pipeline, GL_VERTEX_SHADER_BIT, vertexProgram);
    glUseProgramStages(pipeline, GL_FRAGMENT_SHADER_BIT, fragmentProgram);
    glBindProgramPipeline(pipeline);
    glUseProgram(0);
    ASSERT_GL_NO_ERROR();

    constexpr std::array<GLuint, 6> kInitialVertexOutput = {};
    GLBuffer vertexOutputBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexOutputBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(kInitialVertexOutput),
                 kInitialVertexOutput.data(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vertexOutputBuffer);

    constexpr GLfloat kFragmentScale = 0.5f;
    GLBuffer fragmentScaleBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, fragmentScaleBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(kFragmentScale), &kFragmentScale, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, fragmentScaleBuffer);

    drawQuadPPO(vertexProgram, "position", 0.0f, 1.0f);
    ASSERT_GL_NO_ERROR();
    EXPECT_PIXEL_COLOR_EQ(getWindowWidth() / 2, getWindowHeight() / 2, GLColor(128, 0, 0, 128));

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexOutputBuffer);
    const auto *vertexOutputValues = static_cast<const GLuint *>(glMapBufferRange(
        GL_SHADER_STORAGE_BUFFER, 0, sizeof(kInitialVertexOutput), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, vertexOutputValues);
    for (GLuint i = 0; i < kInitialVertexOutput.size(); ++i)
    {
        EXPECT_EQ(i + 17u, vertexOutputValues[i]);
    }
    EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));
    EXPECT_GL_NO_ERROR();

    glProgramUniform1f(fragmentProgram, intensityLocation, 0.0f);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    drawQuadPPO(vertexProgram, "position", 0.0f, 1.0f);
    ASSERT_GL_NO_ERROR();
    EXPECT_PIXEL_COLOR_EQ(getWindowWidth() / 2, getWindowHeight() / 2, GLColor(0, 0, 0, 0));
}

TEST_P(ExperimentalGles31MetalComputeTest, AtomicAddSharedAndStorageBuffer)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x = 8) in;
layout(std430, binding = 0) buffer Counters
{
    uint deviceCounter;
    uint sharedResult;
    uint deviceResult;
    int sharedCompareSwapResult;
    uint sharedCompareSwapSuccessCount;
} counters;
shared uint sharedCounter;
shared int sharedCompareSwap;
void incrementSharedCounter()
{
    atomicAdd(sharedCounter, 1u);
}
void main()
{
    if (gl_LocalInvocationIndex == 0u)
    {
        sharedCounter = 0u;
        sharedCompareSwap = 0;
        counters.deviceCounter = 0u;
    }
    barrier();

    incrementSharedCounter();
    atomicAdd(counters.deviceCounter, 1u);
    int previousCompareSwap = atomicCompSwap(
        sharedCompareSwap, 0, int(gl_LocalInvocationIndex + 1u));
    if (previousCompareSwap == 0)
    {
        atomicAdd(counters.sharedCompareSwapSuccessCount, 1u);
    }
    barrier();

    if (gl_LocalInvocationIndex == 0u)
    {
        counters.sharedResult = sharedCounter;
        counters.deviceResult = counters.deviceCounter;
        counters.sharedCompareSwapResult = sharedCompareSwap;
    }
})";

    GLProgram program;
    program.makeEmpty();
    GLShader shader(GL_COMPUTE_SHADER);
    const char *source = kCS;
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    ASSERT_GL_NO_ERROR();
    GLint compileStatus = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compileStatus);
    ASSERT_EQ(GL_TRUE, compileStatus);
    glAttachShader(program, shader);
    ASSERT_GL_NO_ERROR();
    glLinkProgram(program);
    ASSERT_GL_NO_ERROR();
    GLint linkStatus = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
    ASSERT_GL_NO_ERROR();
    if (linkStatus != GL_TRUE)
    {
        GLint infoLogLength = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength);
        std::string infoLog(static_cast<size_t>(infoLogLength), '\0');
        GLsizei written = 0;
        glGetProgramInfoLog(program, infoLogLength, &written, infoLog.data());
        FAIL() << infoLog;
    }

    const std::array<GLuint, 5> initialValues = {};
    GLBuffer buffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(initialValues), initialValues.data(),
                 GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer);
    glUseProgram(program);
    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    const auto *values = static_cast<const GLuint *>(glMapBufferRange(
        GL_SHADER_STORAGE_BUFFER, 0, sizeof(initialValues), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, values);
    EXPECT_EQ(8u, values[1]);
    EXPECT_EQ(8u, values[2]);
    EXPECT_GE(values[3], 1u);
    EXPECT_LE(values[3], 8u);
    EXPECT_EQ(1u, values[4]);
    EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));
    EXPECT_GL_NO_ERROR();
}

TEST_P(ExperimentalGles31MetalComputeTest, AtomicIntegerOperatorsStorageBuffer)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x = 8) in;
layout(std430, binding = 0) buffer Results
{
    uint sum;
    uint minimum;
    uint maximum;
    uint bitAnd;
    uint bitOr;
    uint bitXor;
    uint exchange;
    uint compareSwap;
    uint compareSwapSuccessCount;
} results;
void main()
{
    if (gl_LocalInvocationIndex == 0u)
    {
        results.sum = 0u;
        results.minimum = 0xffffffffu;
        results.maximum = 0u;
        results.bitAnd = 0xffffffffu;
        results.bitOr = 0u;
        results.bitXor = 0u;
        results.exchange = 0u;
        results.compareSwap = 0u;
        results.compareSwapSuccessCount = 0u;
    }
    barrier();

    uint value = gl_LocalInvocationIndex + 1u;
    atomicAdd(results.sum, value);
    atomicMin(results.minimum, value);
    atomicMax(results.maximum, value);
    atomicAnd(results.bitAnd, value);
    atomicOr(results.bitOr, value);
    atomicXor(results.bitXor, value);
    atomicExchange(results.exchange, value);
    uint previousCompareSwap = atomicCompSwap(results.compareSwap, 0u, value);
    if (previousCompareSwap == 0u)
    {
        atomicAdd(results.compareSwapSuccessCount, 1u);
    }
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);

    const std::array<GLuint, 9> initialValues = {};
    GLBuffer buffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(initialValues), initialValues.data(),
                 GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer);
    glUseProgram(program);
    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    const auto *values = static_cast<const GLuint *>(glMapBufferRange(
        GL_SHADER_STORAGE_BUFFER, 0, sizeof(initialValues), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, values);
    EXPECT_EQ(36u, values[0]);
    EXPECT_EQ(1u, values[1]);
    EXPECT_EQ(8u, values[2]);
    EXPECT_EQ(0u, values[3]);
    EXPECT_EQ(15u, values[4]);
    EXPECT_EQ(8u, values[5]);
    EXPECT_GE(values[6], 1u);
    EXPECT_LE(values[6], 8u);
    EXPECT_GE(values[7], 1u);
    EXPECT_LE(values[7], 8u);
    EXPECT_EQ(1u, values[8]);
    EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));
    EXPECT_GL_NO_ERROR();
}

TEST_P(ExperimentalGles31MetalComputeTest, AtomicCounterBuffer)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x = 8) in;
layout(binding = 0, offset = 0) uniform atomic_uint counter0;
layout(binding = 1, offset = 0) uniform atomic_uint counter1;
layout(std430, binding = 1) buffer Results
{
    uint values[6];
} results;
void main()
{
    atomicCounterIncrement(counter0);
    atomicCounterIncrement(counter1);
    barrier();
    if (gl_LocalInvocationIndex == 0u)
    {
        results.values[0] = atomicCounter(counter0);
        results.values[1] = atomicCounterDecrement(counter0);
        results.values[2] = atomicCounter(counter0);
        results.values[3] = atomicCounter(counter1);
        results.values[4] = atomicCounterDecrement(counter1);
        results.values[5] = atomicCounter(counter1);
    }
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);

    const std::array<GLuint, 3> initialCounterValues0 = {0xAAAAAAAAu, 5u, 0xBBBBBBBBu};
    GLBuffer counterBuffer0;
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, counterBuffer0);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(initialCounterValues0),
                 initialCounterValues0.data(), GL_DYNAMIC_DRAW);
    glBindBufferRange(GL_ATOMIC_COUNTER_BUFFER, 0, counterBuffer0, sizeof(GLuint), sizeof(GLuint));

    const std::array<GLuint, 3> initialCounterValues1 = {0xCCCCCCCCu, 9u, 0xDDDDDDDDu};
    GLBuffer counterBuffer1;
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, counterBuffer1);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(initialCounterValues1),
                 initialCounterValues1.data(), GL_DYNAMIC_DRAW);
    glBindBufferRange(GL_ATOMIC_COUNTER_BUFFER, 1, counterBuffer1, sizeof(GLuint), sizeof(GLuint));

    const std::array<GLuint, 6> initialResults = {};
    GLBuffer resultBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, resultBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(initialResults), initialResults.data(),
                 GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, resultBuffer);

    glUseProgram(program);
    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();
    glMemoryBarrier(GL_ATOMIC_COUNTER_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT |
                    GL_BUFFER_UPDATE_BARRIER_BIT);

    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, counterBuffer0);
    const auto *counterValues0 = static_cast<const GLuint *>(glMapBufferRange(
        GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(initialCounterValues0), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, counterValues0);
    EXPECT_EQ(0xAAAAAAAAu, counterValues0[0]);
    EXPECT_EQ(12u, counterValues0[1]);
    EXPECT_EQ(0xBBBBBBBBu, counterValues0[2]);
    EXPECT_GL_TRUE(glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER));

    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, counterBuffer1);
    const auto *counterValues1 = static_cast<const GLuint *>(glMapBufferRange(
        GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(initialCounterValues1), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, counterValues1);
    EXPECT_EQ(0xCCCCCCCCu, counterValues1[0]);
    EXPECT_EQ(16u, counterValues1[1]);
    EXPECT_EQ(0xDDDDDDDDu, counterValues1[2]);
    EXPECT_GL_TRUE(glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER));

    const auto *results = static_cast<const GLuint *>(glMapBufferRange(
        GL_SHADER_STORAGE_BUFFER, 0, sizeof(initialResults), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, results);
    EXPECT_EQ(13u, results[0]);
    EXPECT_EQ(12u, results[1]);
    EXPECT_EQ(12u, results[2]);
    EXPECT_EQ(17u, results[3]);
    EXPECT_EQ(16u, results[4]);
    EXPECT_EQ(16u, results[5]);
    EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));
    EXPECT_GL_NO_ERROR();
}

TEST_P(ExperimentalGles31MetalComputeTest, GraphicsStageAtomicCounterBuffers)
{
    GLint maxVertexAtomicCounterBuffers   = 0;
    GLint maxFragmentAtomicCounterBuffers = 0;
    glGetIntegerv(GL_MAX_VERTEX_ATOMIC_COUNTER_BUFFERS, &maxVertexAtomicCounterBuffers);
    glGetIntegerv(GL_MAX_FRAGMENT_ATOMIC_COUNTER_BUFFERS, &maxFragmentAtomicCounterBuffers);
    ASSERT_GT(maxVertexAtomicCounterBuffers, 0);
    ASSERT_GT(maxFragmentAtomicCounterBuffers, 0);

    constexpr char kVS[] = R"(#version 310 es
layout(binding = 0, offset = 0) uniform atomic_uint vertexCounter;
void main()
{
    atomicCounterIncrement(vertexCounter);
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
    gl_PointSize = 1.0;
})";
    constexpr char kFS[] = R"(#version 310 es
precision highp float;
layout(binding = 1, offset = 0) uniform atomic_uint fragmentCounter;
layout(location = 0) out vec4 color;
void main()
{
    atomicCounterIncrement(fragmentCounter);
    color = vec4(1.0, 0.0, 0.0, 1.0);
})";
    ANGLE_GL_PROGRAM(program, kVS, kFS);

    const std::array<GLuint, 4> initialValues = {0xAAAAAAAAu, 0u, 0u, 0xBBBBBBBBu};
    GLBuffer counterBuffer;
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, counterBuffer);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(initialValues), initialValues.data(),
                 GL_DYNAMIC_DRAW);
    glBindBufferRange(GL_ATOMIC_COUNTER_BUFFER, 0, counterBuffer, sizeof(GLuint), sizeof(GLuint));
    glBindBufferRange(GL_ATOMIC_COUNTER_BUFFER, 1, counterBuffer, 2 * sizeof(GLuint),
                      sizeof(GLuint));

    GLTexture colorTexture;
    glBindTexture(GL_TEXTURE_2D, colorTexture);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 1, 1);
    GLFramebuffer framebuffer;
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture, 0);
    ASSERT_GL_FRAMEBUFFER_COMPLETE(GL_FRAMEBUFFER);

    glViewport(0, 0, 1, 1);
    glUseProgram(program);
    glDrawArrays(GL_POINTS, 0, 1);
    EXPECT_GL_NO_ERROR();
    glMemoryBarrier(GL_ATOMIC_COUNTER_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

    std::array<GLubyte, 4> pixel = {};
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel.data());
    EXPECT_GL_NO_ERROR();
    EXPECT_EQ(255u, pixel[0]);
    EXPECT_EQ(0u, pixel[1]);
    EXPECT_EQ(0u, pixel[2]);
    EXPECT_EQ(255u, pixel[3]);

    const auto *values = static_cast<const GLuint *>(glMapBufferRange(
        GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(initialValues), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, values);
    EXPECT_EQ(0xAAAAAAAAu, values[0]);
    EXPECT_EQ(1u, values[1]);
    EXPECT_EQ(1u, values[2]);
    EXPECT_EQ(0xBBBBBBBBu, values[3]);
    EXPECT_GL_TRUE(glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER));
    EXPECT_GL_NO_ERROR();
}

TEST_P(ExperimentalGles31MetalComputeTest, SharedArraysAndStructs)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x = 4) in;
layout(std430, binding = 0) buffer Output
{
    uint values[];
} outputData;
struct SharedValues
{
    uint first;
    uint second;
};
shared uint sharedArray[4];
shared SharedValues sharedValues;
void writeSharedValues(uint index)
{
    sharedArray[index] = index + 10u;
    if (index == 0u)
    {
        sharedValues.first = 42u;
        sharedValues.second = 99u;
    }
}
void main()
{
    writeSharedValues(gl_LocalInvocationIndex);
    barrier();
    if (gl_LocalInvocationIndex == 0u)
    {
        outputData.values[0] = sharedArray[3];
        outputData.values[1] = sharedValues.first;
        outputData.values[2] = sharedValues.second;
    }
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);

    const std::array<GLuint, 3> initialValues = {};
    GLBuffer buffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(initialValues), initialValues.data(),
                 GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer);
    glUseProgram(program);
    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    const auto *values = static_cast<const GLuint *>(glMapBufferRange(
        GL_SHADER_STORAGE_BUFFER, 0, sizeof(initialValues), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, values);
    EXPECT_EQ(13u, values[0]);
    EXPECT_EQ(42u, values[1]);
    EXPECT_EQ(99u, values[2]);
    EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));
    EXPECT_GL_NO_ERROR();
}

TEST_P(ExperimentalGles31MetalComputeTest, NestedSharedStructArrays)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x = 4) in;
layout(std430, binding = 0) buffer Output
{
    uint values[];
} outputData;
struct SharedInner
{
    uint scalar;
    uvec2 pair;
};
struct SharedOuter
{
    SharedInner inner[2];
    uint tail[2];
};
shared SharedOuter sharedValues[2];
void writeSharedValues(uint index)
{
    uint outerIndex = index / 2u;
    uint innerIndex = index % 2u;
    sharedValues[outerIndex].inner[innerIndex].scalar = index + 10u;
    sharedValues[outerIndex].inner[innerIndex].pair = uvec2(index + 20u, index + 30u);
    sharedValues[outerIndex].tail[innerIndex] = index + 40u;
}
void main()
{
    uint index = gl_LocalInvocationIndex;
    writeSharedValues(index);
    barrier();
    if (index == 0u)
    {
        outputData.values[0] = sharedValues[0].inner[1].scalar;
        outputData.values[1] = sharedValues[0].inner[1].pair.x;
        outputData.values[2] = sharedValues[0].inner[1].pair.y;
        outputData.values[3] = sharedValues[0].tail[1];
        outputData.values[4] = sharedValues[1].inner[1].scalar;
        outputData.values[5] = sharedValues[1].inner[1].pair.x;
        outputData.values[6] = sharedValues[1].inner[1].pair.y;
        outputData.values[7] = sharedValues[1].tail[1];
    }
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);

    const std::array<GLuint, 8> initialValues = {};
    GLBuffer buffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(initialValues), initialValues.data(),
                 GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer);
    glUseProgram(program);
    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    const auto *values = static_cast<const GLuint *>(glMapBufferRange(
        GL_SHADER_STORAGE_BUFFER, 0, sizeof(initialValues), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, values);
    EXPECT_EQ(11u, values[0]);
    EXPECT_EQ(21u, values[1]);
    EXPECT_EQ(31u, values[2]);
    EXPECT_EQ(41u, values[3]);
    EXPECT_EQ(13u, values[4]);
    EXPECT_EQ(23u, values[5]);
    EXPECT_EQ(33u, values[6]);
    EXPECT_EQ(43u, values[7]);
    EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));
    EXPECT_GL_NO_ERROR();
}

TEST_P(ExperimentalGles31MetalComputeTest, SharedMemoryLimitCountsAllVariables)
{
    GLint maxSharedMemorySize = 0;
    glGetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &maxSharedMemorySize);
    ASSERT_GT(maxSharedMemorySize, 0);

    const GLint maxSharedUintCount     = maxSharedMemorySize / sizeof(GLuint);
    const GLint halfMaxSharedUintCount = maxSharedUintCount / 2;

    std::ostringstream shaderSource;
    shaderSource << "#version 310 es\n"
                 << "layout(local_size_x = 1) in;\n"
                 << "shared uint first[" << halfMaxSharedUintCount << "];\n"
                 << "shared uint second[" << (maxSharedUintCount - halfMaxSharedUintCount)
                 << "];\n"
                 << "shared uint extra;\n"
                 << "void main() { first[0] = second[0] = extra = 0u; barrier(); }\n";

    const char kOverflowShader[] = R"(#version 310 es
layout(local_size_x = 1) in;
shared uint oversized[1073741825];
void main() { oversized[0] = 0u; barrier(); }
)";

    auto expectSharedMemoryLimitError = [](const std::string &sourceString) {
        const char *source = sourceString.c_str();
        GLShader shader(GL_COMPUTE_SHADER);
        glShaderSource(shader, 1, &source, nullptr);
        glCompileShader(shader);

        GLint compileStatus = GL_TRUE;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &compileStatus);
        EXPECT_EQ(GL_FALSE, compileStatus);

        GLint infoLogLength = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength);
        ASSERT_GT(infoLogLength, 0);
        std::string infoLog(infoLogLength, '\0');
        glGetShaderInfoLog(shader, infoLogLength, nullptr, infoLog.data());
        EXPECT_NE(std::string::npos,
                  infoLog.find("Shared memory size exceeds GL_MAX_COMPUTE_SHARED_MEMORY_SIZE"));
    };

    expectSharedMemoryLimitError(shaderSource.str());
    expectSharedMemoryLimitError(kOverflowShader);
}

TEST_P(ExperimentalGles31MetalComputeTest, ImageLoadStore)
{
    constexpr char kCS[]     = R"(#version 310 es
layout(local_size_x = 1) in;
layout(r32ui, binding = 0) uniform highp writeonly uimage2D image;
layout(std430, binding = 0) buffer Output
{
    uint values[];
} outputData;
void main()
{
    ivec2 coord = ivec2(gl_GlobalInvocationID.x, 0);
    uint value = 109u + gl_GlobalInvocationID.x;
    outputData.values[gl_GlobalInvocationID.x] = value;
    imageStore(image, coord, uvec4(value));
})";
    constexpr char kReadCS[] = R"(#version 310 es
layout(local_size_x = 1) in;
layout(r32ui, binding = 0) uniform highp readonly uimage2D image;
layout(std430, binding = 0) buffer Output
{
    uint values[];
} outputData;
void main()
{
    ivec2 coord = ivec2(gl_GlobalInvocationID.x, 0);
    outputData.values[gl_GlobalInvocationID.x] = imageLoad(image, coord).x;
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    ANGLE_GL_COMPUTE_PROGRAM(readProgram, kReadCS);

    constexpr GLuint kWidth = 8;
    GLBuffer buffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, kWidth * sizeof(GLuint), nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer);

    GLTexture texture;
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, kWidth, 1);
    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
    glUseProgram(program);
    glDispatchCompute(kWidth, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    const auto *bufferValues = static_cast<const GLuint *>(
        glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, kWidth * sizeof(GLuint), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, bufferValues);
    for (GLuint i = 0; i < kWidth; ++i)
    {
        EXPECT_EQ(109 + i, bufferValues[i]);
    }
    EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));

    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI);
    glUseProgram(readProgram);
    glDispatchCompute(kWidth, 1, 1);
    EXPECT_GL_NO_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    const auto *outputValues = static_cast<const GLuint *>(
        glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, kWidth * sizeof(GLuint), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, outputValues);
    for (GLuint i = 0; i < kWidth; ++i)
    {
        EXPECT_EQ(109 + i, outputValues[i]);
    }
    EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));
}

TEST_P(ExperimentalGles31MetalComputeTest, ShaderImageAtomicIntegerOperations)
{
    ASSERT_TRUE(IsGLExtensionEnabled("GL_OES_shader_image_atomic"));

    constexpr char kCS[] = R"(#version 310 es
#extension GL_OES_shader_image_atomic : require
layout(local_size_x = 1) in;
layout(r32i, binding = 0) uniform highp iimage2D signedImage;
layout(r32ui, binding = 1) uniform highp uimage2D unsignedImage;
layout(std430, binding = 0) buffer Output
{
    uint values[34];
} outputData;
void main()
{
    outputData.values[0] = uint(imageAtomicAdd(signedImage, ivec2(0, 0), -2));
    outputData.values[1] = uint(imageAtomicMin(signedImage, ivec2(1, 0), -7));
    outputData.values[2] = uint(imageAtomicMax(signedImage, ivec2(2, 0), 9));
    outputData.values[3] = uint(imageAtomicAnd(signedImage, ivec2(3, 0), 0x3c));
    outputData.values[4] = uint(imageAtomicOr(signedImage, ivec2(4, 0), 0x0c));
    outputData.values[5] = uint(imageAtomicXor(signedImage, ivec2(5, 0), 0x0f));
    outputData.values[6] = uint(imageAtomicExchange(signedImage, ivec2(6, 0), -11));
    outputData.values[7] = uint(imageAtomicCompSwap(signedImage, ivec2(7, 0), 42, -15));

    outputData.values[8] = imageAtomicAdd(unsignedImage, ivec2(0, 0), 7u);
    outputData.values[9] = imageAtomicMin(unsignedImage, ivec2(1, 0), 2u);
    outputData.values[10] = imageAtomicMax(unsignedImage, ivec2(2, 0), 9u);
    outputData.values[11] = imageAtomicAnd(unsignedImage, ivec2(3, 0), 0x3cu);
    outputData.values[12] = imageAtomicOr(unsignedImage, ivec2(4, 0), 0x0cu);
    outputData.values[13] = imageAtomicXor(unsignedImage, ivec2(5, 0), 0x0fu);
    outputData.values[14] = imageAtomicExchange(unsignedImage, ivec2(6, 0), 23u);
    outputData.values[15] = imageAtomicCompSwap(unsignedImage, ivec2(7, 0), 42u, 99u);
    outputData.values[32] = uint(imageAtomicAdd(signedImage, ivec2(-1, 0), 1));
    outputData.values[33] = imageAtomicAdd(unsignedImage, ivec2(8, 0), 1u);

    for (int i = 0; i < 8; ++i)
    {
        outputData.values[16 + i] = uint(imageLoad(signedImage, ivec2(i, 0)).r);
        outputData.values[24 + i] = imageLoad(unsignedImage, ivec2(i, 0)).r;
    }
})";
    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);

    constexpr std::array<GLint, 8> kSignedInitial    = {5, 5, 5, 0xf0, 0x30, 0xff, 17, 42};
    constexpr std::array<GLuint, 8> kUnsignedInitial = {5u, 5u, 5u, 0xf0u, 0x30u, 0xffu, 17u, 42u};
    GLTexture signedImage;
    glBindTexture(GL_TEXTURE_2D, signedImage);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32I, kSignedInitial.size(), 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kSignedInitial.size(), 1, GL_RED_INTEGER, GL_INT,
                    kSignedInitial.data());
    GLTexture unsignedImage;
    glBindTexture(GL_TEXTURE_2D, unsignedImage);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, kUnsignedInitial.size(), 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kUnsignedInitial.size(), 1, GL_RED_INTEGER,
                    GL_UNSIGNED_INT, kUnsignedInitial.data());
    ASSERT_GL_NO_ERROR();

    GLBuffer outputBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 34 * sizeof(GLuint), nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, outputBuffer);
    glBindImageTexture(0, signedImage, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32I);
    glBindImageTexture(1, unsignedImage, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
    glUseProgram(program);
    glDispatchCompute(1, 1, 1);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    constexpr std::array<GLuint, 34> kExpected = {
        5u,
        5u,
        5u,
        0xf0u,
        0x30u,
        0xffu,
        17u,
        42u,
        5u,
        5u,
        5u,
        0xf0u,
        0x30u,
        0xffu,
        17u,
        42u,
        3u,
        static_cast<GLuint>(-7),
        9u,
        0x30u,
        0x3cu,
        0xf0u,
        static_cast<GLuint>(-11),
        static_cast<GLuint>(-15),
        12u,
        2u,
        9u,
        0x30u,
        0x3cu,
        0xf0u,
        23u,
        99u,
        0u,
        0u,
    };
    const auto *actual = static_cast<const GLuint *>(
        glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeof(kExpected), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, actual);
    for (size_t i = 0; i < kExpected.size(); ++i)
    {
        EXPECT_EQ(kExpected[i], actual[i]) << "atomic result/final texel index " << i;
    }
    EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));
    EXPECT_GL_NO_ERROR();
}

TEST_P(ExperimentalGles31MetalComputeTest, ShaderImageAtomicExchangeR32F)
{
    ASSERT_TRUE(IsGLExtensionEnabled("GL_OES_shader_image_atomic"));

    constexpr char kCS[] = R"(#version 310 es
#extension GL_OES_shader_image_atomic : require
layout(local_size_x = 1) in;
layout(r32f, binding = 0) uniform highp image2D floatImage;
layout(std430, binding = 0) buffer Output
{
    vec2 values;
} outputData;
void main()
{
    outputData.values.x = imageAtomicExchange(floatImage, ivec2(0), 2.5);
    outputData.values.y = imageLoad(floatImage, ivec2(0)).r;
})";
    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);

    constexpr GLfloat kInitialValue = 1.25f;
    GLTexture floatImage;
    glBindTexture(GL_TEXTURE_2D, floatImage);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, 1, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RED, GL_FLOAT, &kInitialValue);
    ASSERT_GL_NO_ERROR();

    GLBuffer outputBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 2 * sizeof(GLfloat), nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, outputBuffer);
    glBindImageTexture(0, floatImage, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F);
    glUseProgram(program);
    glDispatchCompute(1, 1, 1);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    const auto *actual = static_cast<const GLfloat *>(
        glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 2 * sizeof(GLfloat), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, actual);
    EXPECT_FLOAT_EQ(1.25f, actual[0]);
    EXPECT_FLOAT_EQ(2.5f, actual[1]);
    EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));
    EXPECT_GL_NO_ERROR();
}

TEST_P(ExperimentalGles31MetalComputeTest, ShaderImageAtomicExchangeR32FReturnImage)
{
    ASSERT_TRUE(IsGLExtensionEnabled("GL_OES_shader_image_atomic"));

    constexpr char kAtomicCS[]   = R"(#version 310 es
#extension GL_OES_shader_image_atomic : require
layout(local_size_x = 5) in;
layout(r32f, binding = 0) coherent uniform highp image2D sourceImage;
layout(r32f, binding = 1) writeonly uniform highp image2D returnImage;
void main()
{
    int x = int(gl_GlobalInvocationID.x);
    float value = float(x * 4096);
    imageStore(returnImage, ivec2(x, 0),
               vec4(imageAtomicExchange(sourceImage, ivec2(0), value)));
})";
    constexpr char kReadbackCS[] = R"(#version 310 es
layout(local_size_x = 5) in;
uniform highp sampler2D returnTexture;
layout(std430, binding = 0) buffer Output
{
    float values[5];
} outputData;
void main()
{
    int x = int(gl_GlobalInvocationID.x);
    outputData.values[x] = texelFetch(returnTexture, ivec2(x, 0), 0).r;
})";
    ANGLE_GL_COMPUTE_PROGRAM(atomicProgram, kAtomicCS);
    ANGLE_GL_COMPUTE_PROGRAM(readbackProgram, kReadbackCS);

    constexpr GLfloat kInitialValue = 18.0f;
    GLTexture sourceImage;
    glBindTexture(GL_TEXTURE_2D, sourceImage);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, 1, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RED, GL_FLOAT, &kInitialValue);

    GLTexture returnImage;
    glBindTexture(GL_TEXTURE_2D, returnImage);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, 5, 1);
    ASSERT_GL_NO_ERROR();

    GLBuffer outputBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 5 * sizeof(GLfloat), nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, outputBuffer);

    glBindImageTexture(0, sourceImage, 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32F);
    glBindImageTexture(1, returnImage, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);
    glUseProgram(atomicProgram);
    glDispatchCompute(1, 1, 1);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, returnImage);
    glUseProgram(readbackProgram);
    glUniform1i(glGetUniformLocation(readbackProgram, "returnTexture"), 0);
    glDispatchCompute(1, 1, 1);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    const auto *actual = static_cast<const GLfloat *>(
        glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 5 * sizeof(GLfloat), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, actual);

    const auto initialValue = std::find(actual, actual + 5, kInitialValue);
    ASSERT_NE(actual + 5, initialValue)
        << "return values: " << actual[0] << ", " << actual[1] << ", " << actual[2] << ", "
        << actual[3] << ", " << actual[4];
    EXPECT_EQ(1, std::count(actual, actual + 5, kInitialValue));
    constexpr std::array<GLfloat, 5> kAtomicValues = {0.0f, 4096.0f, 8192.0f, 12288.0f, 16384.0f};
    std::array<bool, kAtomicValues.size()> seenValues = {};
    size_t returnedAtomicValues                       = 0;
    for (size_t i = 0; i < 5; ++i)
    {
        if (actual[i] == kInitialValue)
        {
            continue;
        }
        const auto value = std::find(kAtomicValues.begin(), kAtomicValues.end(), actual[i]);
        ASSERT_NE(kAtomicValues.end(), value) << "unexpected atomic return value " << actual[i];
        const size_t valueIndex = static_cast<size_t>(value - kAtomicValues.begin());
        EXPECT_FALSE(seenValues[valueIndex]) << "duplicate atomic return value " << actual[i];
        seenValues[valueIndex] = true;
        ++returnedAtomicValues;
    }
    EXPECT_EQ(4u, returnedAtomicValues);
    EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));
    EXPECT_GL_NO_ERROR();
}

TEST_P(ExperimentalGles31MetalComputeTest, ShaderImageAtomicLayeredImageTypes)
{
    ASSERT_TRUE(IsGLExtensionEnabled("GL_OES_shader_image_atomic"));

    constexpr char kCS[] = R"(#version 310 es
#extension GL_OES_shader_image_atomic : require
layout(local_size_x = 1) in;
layout(r32ui, binding = 0) uniform highp uimage2DArray arrayImage;
layout(r32ui, binding = 1) uniform highp uimage3D volumeImage;
layout(r32ui, binding = 2) uniform highp uimageCube cubeImage;
layout(std430, binding = 0) buffer Output
{
    uint values[6];
} outputData;
void main()
{
    ivec3 arrayCoord = ivec3(0, 0, 1);
    ivec3 image3DCoord = ivec3(0, 0, 1);
    ivec3 cubeCoord = ivec3(0, 0, 4);
    outputData.values[0] = imageAtomicAdd(arrayImage, arrayCoord, 10u);
    outputData.values[1] = imageLoad(arrayImage, arrayCoord).r;
    outputData.values[2] = imageAtomicAdd(volumeImage, image3DCoord, 10u);
    outputData.values[3] = imageLoad(volumeImage, image3DCoord).r;
    outputData.values[4] = imageAtomicAdd(cubeImage, cubeCoord, 10u);
    outputData.values[5] = imageLoad(cubeImage, cubeCoord).r;
})";
    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);

    GLTexture arrayImage;
    glBindTexture(GL_TEXTURE_2D_ARRAY, arrayImage);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_R32UI, 1, 1, 2);
    constexpr std::array<GLuint, 2> kArrayInitial = {3u, 4u};
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, 1, 1, kArrayInitial.size(), GL_RED_INTEGER,
                    GL_UNSIGNED_INT, kArrayInitial.data());

    GLTexture image3D;
    glBindTexture(GL_TEXTURE_3D, image3D);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_R32UI, 1, 1, 2);
    constexpr std::array<GLuint, 2> kImage3DInitial = {5u, 6u};
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, 1, 1, kImage3DInitial.size(), GL_RED_INTEGER,
                    GL_UNSIGNED_INT, kImage3DInitial.data());

    GLTexture cubeImage;
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubeImage);
    glTexStorage2D(GL_TEXTURE_CUBE_MAP, 1, GL_R32UI, 1, 1);
    constexpr std::array<GLuint, 6> kCubeInitial = {0u, 0u, 0u, 0u, 7u, 0u};
    for (size_t face = 0; face < kCubeInitial.size(); ++face)
    {
        glTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + static_cast<GLenum>(face), 0, 0, 0, 1, 1,
                        GL_RED_INTEGER, GL_UNSIGNED_INT, &kCubeInitial[face]);
    }
    ASSERT_GL_NO_ERROR();

    GLBuffer outputBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 6 * sizeof(GLuint), nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, outputBuffer);
    glBindImageTexture(0, arrayImage, 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32UI);
    glBindImageTexture(1, image3D, 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32UI);
    glBindImageTexture(2, cubeImage, 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32UI);
    glUseProgram(program);
    glDispatchCompute(1, 1, 1);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    constexpr std::array<GLuint, 6> kExpected = {4u, 14u, 6u, 16u, 7u, 17u};
    const auto *actual                        = static_cast<const GLuint *>(
        glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeof(kExpected), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, actual);
    for (size_t i = 0; i < kExpected.size(); ++i)
    {
        EXPECT_EQ(kExpected[i], actual[i]) << "layered image result index " << i;
    }
    EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));
    EXPECT_GL_NO_ERROR();
}

TEST_P(ExperimentalGles31MetalComputeTest, ShaderImageAtomicNonLayered3DSlice)
{
    ASSERT_TRUE(IsGLExtensionEnabled("GL_OES_shader_image_atomic"));

    constexpr char kCS[] = R"(#version 310 es
#extension GL_OES_shader_image_atomic : require
layout(local_size_x = 1) in;
layout(r32ui, binding = 0) uniform highp uimage2D image;
layout(std430, binding = 0) buffer Output
{
    uint values[2];
} outputData;
void main()
{
    outputData.values[0] = imageAtomicAdd(image, ivec2(0), 13u);
    outputData.values[1] = imageLoad(image, ivec2(0)).r;
})";
    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);

    GLTexture texture3D;
    glBindTexture(GL_TEXTURE_3D, texture3D);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_R32UI, 1, 1, 2);
    constexpr std::array<GLuint, 2> kInitialValues = {23u, 29u};
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, 1, 1, kInitialValues.size(), GL_RED_INTEGER,
                    GL_UNSIGNED_INT, kInitialValues.data());
    ASSERT_GL_NO_ERROR();

    GLBuffer outputBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 2 * sizeof(GLuint), nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, outputBuffer);
    glBindImageTexture(0, texture3D, 0, GL_FALSE, 1, GL_READ_WRITE, GL_R32UI);
    glUseProgram(program);
    glDispatchCompute(1, 1, 1);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    constexpr std::array<GLuint, 2> kExpected = {29u, 42u};
    const auto *actual                        = static_cast<const GLuint *>(
        glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeof(kExpected), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, actual);
    EXPECT_EQ(kExpected[0], actual[0]);
    EXPECT_EQ(kExpected[1], actual[1]);
    EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));
    EXPECT_GL_NO_ERROR();
}

TEST_P(ExperimentalGles31MetalComputeTest, ShaderImageAtomicGraphicsStages)
{
    ASSERT_TRUE(IsGLExtensionEnabled("GL_OES_shader_image_atomic"));

    constexpr char kAtomicVS[] = R"(#version 310 es
#extension GL_OES_shader_image_atomic : require
layout(r32ui, binding = 0) uniform highp uimage2D image;
layout(std430, binding = 0) buffer Output
{
    uint values[2];
} outputData;
void main()
{
    outputData.values[0] = imageAtomicAdd(image, ivec2(0), 2u);
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
    gl_PointSize = 1.0;
})";
    constexpr char kAtomicFS[] = R"(#version 310 es
#extension GL_OES_shader_image_atomic : require
precision highp float;
layout(r32ui, binding = 0) uniform highp uimage2D image;
layout(std430, binding = 0) buffer Output
{
    uint values[2];
} outputData;
out vec4 color;
void main()
{
    outputData.values[1] = imageAtomicAdd(image, ivec2(0), 3u);
    color = vec4(0.0, 0.0, 0.0, 1.0);
})";
    constexpr char kPlainVS[]  = R"(#version 310 es
void main()
{
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
    gl_PointSize = 1.0;
})";
    constexpr char kPlainFS[]  = R"(#version 310 es
precision highp float;
out vec4 color;
void main()
{
    color = vec4(0.0, 0.0, 0.0, 1.0);
})";
    ANGLE_GL_PROGRAM(vertexAtomicProgram, kAtomicVS, kPlainFS);
    ANGLE_GL_PROGRAM(fragmentAtomicProgram, kPlainVS, kAtomicFS);

    GLTexture image;
    glBindTexture(GL_TEXTURE_2D, image);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, 1, 1);
    constexpr GLuint kInitialValue = 10u;
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, &kInitialValue);
    glBindImageTexture(0, image, 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32UI);

    GLBuffer outputBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 2 * sizeof(GLuint), nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, outputBuffer);

    GLTexture framebufferTexture;
    glBindTexture(GL_TEXTURE_2D, framebufferTexture);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 1, 1);
    GLFramebuffer framebuffer;
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, framebufferTexture,
                           0);
    ASSERT_GL_FRAMEBUFFER_COMPLETE(GL_FRAMEBUFFER);
    glViewport(0, 0, 1, 1);

    glUseProgram(vertexAtomicProgram);
    glDrawArrays(GL_POINTS, 0, 1);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    glUseProgram(fragmentAtomicProgram);
    glDrawArrays(GL_POINTS, 0, 1);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    const auto *actual = static_cast<const GLuint *>(
        glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 2 * sizeof(GLuint), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, actual);
    EXPECT_EQ(10u, actual[0]);
    EXPECT_EQ(12u, actual[1]);
    EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));
    EXPECT_GL_NO_ERROR();
}

TEST_P(ExperimentalGles31MetalComputeTest, ShaderImageAtomicConcurrentAdd)
{
    ASSERT_TRUE(IsGLExtensionEnabled("GL_OES_shader_image_atomic"));

    constexpr char kAtomicCS[] = R"(#version 310 es
#extension GL_OES_shader_image_atomic : require
layout(local_size_x = 64) in;
layout(r32ui, binding = 0) uniform highp uimage2D image;
layout(std430, binding = 0) buffer Output
{
    uint values[65];
} outputData;
void main()
{
    uint invocation = gl_LocalInvocationIndex;
    outputData.values[invocation] = imageAtomicAdd(image, ivec2(0), 1u);
})";
    constexpr char kReadCS[]   = R"(#version 310 es
layout(local_size_x = 1) in;
layout(r32ui, binding = 0) uniform highp readonly uimage2D image;
layout(std430, binding = 0) buffer Output
{
    uint values[65];
} outputData;
void main()
{
    outputData.values[64] = imageLoad(image, ivec2(0)).r;
})";
    ANGLE_GL_COMPUTE_PROGRAM(atomicProgram, kAtomicCS);
    ANGLE_GL_COMPUTE_PROGRAM(readProgram, kReadCS);

    GLTexture image;
    glBindTexture(GL_TEXTURE_2D, image);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, 1, 1);
    constexpr GLuint kInitialValue = 0;
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, &kInitialValue);
    ASSERT_GL_NO_ERROR();

    GLBuffer outputBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 65 * sizeof(GLuint), nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, outputBuffer);
    glBindImageTexture(0, image, 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32UI);
    glUseProgram(atomicProgram);
    glDispatchCompute(1, 1, 1);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    glBindImageTexture(0, image, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32UI);
    glUseProgram(readProgram);
    glDispatchCompute(1, 1, 1);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    const auto *actual = static_cast<const GLuint *>(
        glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 65 * sizeof(GLuint), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, actual);
    std::array<bool, 64> seen = {};
    for (GLuint i = 0; i < seen.size(); ++i)
    {
        ASSERT_LT(actual[i], seen.size());
        EXPECT_FALSE(seen[actual[i]]) << "duplicate atomic return value " << actual[i];
        seen[actual[i]] = true;
    }
    EXPECT_EQ(64u, actual[64]);
    EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));
    EXPECT_GL_NO_ERROR();
}

TEST_P(ExperimentalGles31MetalComputeTest, ShaderImageStoreBarrierThenAtomicAdd)
{
    ASSERT_TRUE(IsGLExtensionEnabled("GL_OES_shader_image_atomic"));

    constexpr char kStoreAndAddCS[] = R"(#version 310 es
#extension GL_OES_shader_image_atomic : require
layout(local_size_x = 8) in;
layout(r32ui, binding = 0) uniform highp uimage2D image;
void main()
{
    if (gl_LocalInvocationIndex == 0u)
        imageStore(image, ivec2(0), uvec4(0u));
    barrier();
    imageAtomicAdd(image, ivec2(0), 1u);
})";
    constexpr char kReadCS[] = R"(#version 310 es
layout(local_size_x = 1) in;
layout(r32ui, binding = 0) uniform highp readonly uimage2D image;
layout(std430, binding = 0) buffer Output
{
    uint value;
} outputData;
void main()
{
    outputData.value = imageLoad(image, ivec2(0)).r;
})";
    ANGLE_GL_COMPUTE_PROGRAM(storeAndAddProgram, kStoreAndAddCS);
    ANGLE_GL_COMPUTE_PROGRAM(readProgram, kReadCS);

    GLTexture image;
    glBindTexture(GL_TEXTURE_2D, image);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, 1, 1);
    constexpr GLuint kInitialValue = 0;
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    &kInitialValue);
    ASSERT_GL_NO_ERROR();

    GLBuffer outputBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, outputBuffer);
    glBindImageTexture(0, image, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
    glUseProgram(storeAndAddProgram);
    glDispatchCompute(1, 1, 1);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    glBindImageTexture(0, image, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI);
    glUseProgram(readProgram);
    glDispatchCompute(1, 1, 1);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    const auto *actual = static_cast<const GLuint *>(
        glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, actual);
    EXPECT_EQ(8u, actual[0]);
    EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));
    EXPECT_GL_NO_ERROR();
}

TEST_P(ExperimentalGles31MetalComputeTest, ShaderImageAtomicExchangeR32FLayeredTypes)
{
    ASSERT_TRUE(IsGLExtensionEnabled("GL_OES_shader_image_atomic"));

    constexpr char kCS[] = R"(#version 310 es
#extension GL_OES_shader_image_atomic : require
layout(local_size_x = 1) in;
layout(r32f, binding = 0) uniform highp image2DArray arrayImage;
layout(r32f, binding = 1) uniform highp image3D volumeImage;
layout(r32f, binding = 2) uniform highp imageCube cubeImage;
layout(std430, binding = 0) buffer Output
{
    float values[6];
} outputData;
void main()
{
    ivec3 coord = ivec3(0, 0, 1);
    outputData.values[0] = imageAtomicExchange(arrayImage, coord, 2.5);
    outputData.values[1] = imageLoad(arrayImage, coord).r;
    outputData.values[2] = imageAtomicExchange(volumeImage, coord, 4.5);
    outputData.values[3] = imageLoad(volumeImage, coord).r;
    outputData.values[4] = imageAtomicExchange(cubeImage, ivec3(0, 0, 4), 6.5);
    outputData.values[5] = imageLoad(cubeImage, ivec3(0, 0, 4)).r;
})";
    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);

    GLTexture arrayImage;
    glBindTexture(GL_TEXTURE_2D_ARRAY, arrayImage);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_R32F, 1, 1, 2);
    constexpr std::array<GLfloat, 2> kArrayInitial = {0.25f, 1.25f};
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, 1, 1, kArrayInitial.size(), GL_RED, GL_FLOAT,
                    kArrayInitial.data());

    GLTexture volumeImage;
    glBindTexture(GL_TEXTURE_3D, volumeImage);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_R32F, 1, 1, 2);
    constexpr std::array<GLfloat, 2> kVolumeInitial = {0.5f, 3.5f};
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, 1, 1, kVolumeInitial.size(), GL_RED, GL_FLOAT,
                    kVolumeInitial.data());

    GLTexture cubeImage;
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubeImage);
    glTexStorage2D(GL_TEXTURE_CUBE_MAP, 1, GL_R32F, 1, 1);
    constexpr std::array<GLfloat, 6> kCubeInitial = {0.0f, 0.0f, 0.0f, 0.0f, 5.5f, 0.0f};
    for (size_t face = 0; face < kCubeInitial.size(); ++face)
    {
        glTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + static_cast<GLenum>(face), 0, 0, 0, 1, 1,
                        GL_RED, GL_FLOAT, &kCubeInitial[face]);
    }

    ASSERT_GL_NO_ERROR();

    GLBuffer outputBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 6 * sizeof(GLfloat), nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, outputBuffer);
    glBindImageTexture(0, arrayImage, 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32F);
    glBindImageTexture(1, volumeImage, 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32F);
    glBindImageTexture(2, cubeImage, 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32F);
    glUseProgram(program);
    glDispatchCompute(1, 1, 1);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    constexpr std::array<GLfloat, 6> kExpected = {1.25f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f};
    const auto *actual                         = static_cast<const GLfloat *>(
        glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeof(kExpected), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, actual);
    for (size_t i = 0; i < kExpected.size(); ++i)
    {
        EXPECT_FLOAT_EQ(kExpected[i], actual[i]) << "R32F atomic exchange value index " << i;
    }
    EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));
    EXPECT_GL_NO_ERROR();
}

TEST_P(ExperimentalGles31MetalComputeTest, ShaderImageAtomicExchangeR32FNonLayered3DSlice)
{
    ASSERT_TRUE(IsGLExtensionEnabled("GL_OES_shader_image_atomic"));

    constexpr char kCS[] = R"(#version 310 es
#extension GL_OES_shader_image_atomic : require
layout(local_size_x = 1) in;
layout(r32f, binding = 0) uniform highp image2D sliceImage;
layout(std430, binding = 0) buffer Output
{
    vec2 values;
} outputData;
void main()
{
    outputData.values.x = imageAtomicExchange(sliceImage, ivec2(0), 8.5);
    outputData.values.y = imageLoad(sliceImage, ivec2(0)).r;
})";
    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);

    GLTexture texture3D;
    glBindTexture(GL_TEXTURE_3D, texture3D);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_R32F, 1, 1, 2);
    constexpr std::array<GLfloat, 2> kInitialValues = {6.5f, 7.5f};
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, 1, 1, kInitialValues.size(), GL_RED, GL_FLOAT,
                    kInitialValues.data());
    ASSERT_GL_NO_ERROR();

    GLBuffer outputBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 2 * sizeof(GLfloat), nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, outputBuffer);
    glBindImageTexture(0, texture3D, 0, GL_FALSE, 1, GL_READ_WRITE, GL_R32F);
    glUseProgram(program);
    glDispatchCompute(1, 1, 1);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    const auto *actual = static_cast<const GLfloat *>(
        glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 2 * sizeof(GLfloat), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, actual);
    EXPECT_FLOAT_EQ(7.5f, actual[0]);
    EXPECT_FLOAT_EQ(8.5f, actual[1]);
    EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));
    EXPECT_GL_NO_ERROR();
}

TEST_P(ExperimentalGles31MetalComputeTest, ImageUniformArray)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x = 1) in;
layout(r32ui, binding = 0) writeonly uniform highp uimage2D images[2];
void main()
{
    imageStore(images[0], ivec2(0), uvec4(11u));
    imageStore(images[1], ivec2(0), uvec4(13u));
})";
    GLuint computeShader = CompileShader(GL_COMPUTE_SHADER, kCS);
    ASSERT_NE(0u, computeShader);
    ASSERT_GL_NO_ERROR();
    GLuint program = glCreateProgram();
    ASSERT_NE(0u, program);
    glAttachShader(program, computeShader);
    glLinkProgram(program);
    ASSERT_GL_NO_ERROR();
    GLint linkStatus = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
    if (linkStatus != GL_TRUE)
    {
        GLint infoLogLength = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength);
        std::vector<GLchar> infoLog(static_cast<size_t>(std::max(infoLogLength, 1)));
        glGetProgramInfoLog(program, static_cast<GLsizei>(infoLog.size()), nullptr, infoLog.data());
        FAIL() << "Program link failed: " << infoLog.data();
    }

    GLTexture image0;
    glBindTexture(GL_TEXTURE_2D, image0);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, 1, 1);
    GLTexture image1;
    glBindTexture(GL_TEXTURE_2D, image1);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, 1, 1);
    glBindImageTexture(0, image0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
    glBindImageTexture(1, image1, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);

    glUseProgram(program);
    glDispatchCompute(1, 1, 1);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT);

    GLFramebuffer framebuffer;
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, image0, 0);
    ASSERT_GL_FRAMEBUFFER_COMPLETE(GL_READ_FRAMEBUFFER);
    GLuint firstValue = 0;
    glReadPixels(0, 0, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, &firstValue);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, image1, 0);
    ASSERT_GL_FRAMEBUFFER_COMPLETE(GL_READ_FRAMEBUFFER);
    GLuint secondValue = 0;
    glReadPixels(0, 0, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, &secondValue);
    EXPECT_GL_NO_ERROR();
    EXPECT_EQ(11u, firstValue);
    EXPECT_EQ(13u, secondValue);
}

TEST_P(ExperimentalGles31MetalComputeTest, LayeredImage2DArrayLoadStore)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(r32ui, binding = 0) uniform highp uimage2DArray image;
void main()
{
    ivec3 coord = ivec3(0, 0, int(gl_GlobalInvocationID.z));
    uint value = imageLoad(image, coord).r;
    imageStore(image, coord, uvec4(value + uint(imageSize(image).z)));
})";
    GLuint computeShader = CompileShader(GL_COMPUTE_SHADER, kCS);
    ASSERT_NE(0u, computeShader);
    ASSERT_GL_NO_ERROR();
    GLuint program = glCreateProgram();
    ASSERT_NE(0u, program);
    glAttachShader(program, computeShader);
    glLinkProgram(program);
    ASSERT_GL_NO_ERROR();
    GLint linkStatus = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
    if (linkStatus != GL_TRUE)
    {
        GLint infoLogLength = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength);
        std::vector<GLchar> infoLog(static_cast<size_t>(std::max(infoLogLength, 1)));
        glGetProgramInfoLog(program, static_cast<GLsizei>(infoLog.size()), nullptr, infoLog.data());
        FAIL() << "Program link failed: " << infoLog.data();
    }

    GLTexture image;
    glBindTexture(GL_TEXTURE_2D_ARRAY, image);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_R32UI, 1, 1, 3);
    const std::array<GLuint, 3> initialValues = {3u, 5u, 7u};
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, 1, 1, 3, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    initialValues.data());
    glBindImageTexture(0, image, 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32UI);

    glUseProgram(program);
    glDispatchCompute(1, 1, 3);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT);

    GLFramebuffer framebuffer;
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
    std::array<GLuint, 3> resultValues = {};
    for (GLint layer = 0; layer < 3; ++layer)
    {
        glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, image, 0, layer);
        ASSERT_GL_FRAMEBUFFER_COMPLETE(GL_READ_FRAMEBUFFER);
        glReadPixels(0, 0, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT,
                     &resultValues[static_cast<size_t>(layer)]);
    }
    EXPECT_GL_NO_ERROR();
    EXPECT_EQ((std::array<GLuint, 3>{6u, 8u, 10u}), resultValues);
}

TEST_P(ExperimentalGles31MetalComputeTest, LayeredImage3DLoadStore)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(r32ui, binding = 0) uniform highp uimage3D image;
void main()
{
    ivec3 coord = ivec3(0, 0, int(gl_GlobalInvocationID.z));
    uint value = imageLoad(image, coord).r;
    imageStore(image, coord, uvec4(value + uint(imageSize(image).z)));
})";
    GLuint computeShader = CompileShader(GL_COMPUTE_SHADER, kCS);
    ASSERT_NE(0u, computeShader);
    ASSERT_GL_NO_ERROR();
    GLuint program = glCreateProgram();
    ASSERT_NE(0u, program);
    glAttachShader(program, computeShader);
    glLinkProgram(program);
    ASSERT_GL_NO_ERROR();
    GLint linkStatus = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
    if (linkStatus != GL_TRUE)
    {
        GLint infoLogLength = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength);
        std::vector<GLchar> infoLog(static_cast<size_t>(std::max(infoLogLength, 1)));
        glGetProgramInfoLog(program, static_cast<GLsizei>(infoLog.size()), nullptr, infoLog.data());
        FAIL() << "Program link failed: " << infoLog.data();
    }

    GLTexture image;
    glBindTexture(GL_TEXTURE_3D, image);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_R32UI, 1, 1, 3);
    const std::array<GLuint, 3> initialValues = {3u, 5u, 7u};
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, 1, 1, 3, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    initialValues.data());
    glBindImageTexture(0, image, 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32UI);

    glUseProgram(program);
    glDispatchCompute(1, 1, 3);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT);

    GLFramebuffer framebuffer;
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
    std::array<GLuint, 3> resultValues = {};
    for (GLint layer = 0; layer < 3; ++layer)
    {
        glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, image, 0, layer);
        ASSERT_GL_FRAMEBUFFER_COMPLETE(GL_READ_FRAMEBUFFER);
        glReadPixels(0, 0, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT,
                     &resultValues[static_cast<size_t>(layer)]);
    }
    EXPECT_GL_NO_ERROR();
    EXPECT_EQ((std::array<GLuint, 3>{6u, 8u, 10u}), resultValues);
}

TEST_P(ExperimentalGles31MetalComputeTest, NonLayeredImage2DTexture3DSliceLoadStore)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x = 1) in;
layout(r32ui, binding = 0) readonly uniform highp uimage2D sourceImage;
layout(r32ui, binding = 1) writeonly uniform highp uimage2D destinationImage;
void main()
{
    uint value = imageLoad(sourceImage, ivec2(0)).r;
    imageStore(destinationImage, ivec2(0), uvec4(value + 9u));
})";
    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);

    GLTexture source;
    GLTexture destination;
    glBindTexture(GL_TEXTURE_3D, source);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_R32UI, 1, 1, 3);
    const std::array<GLuint, 3> sourceValues = {17u, 29u, 41u};
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, 1, 1, 3, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    sourceValues.data());
    glBindTexture(GL_TEXTURE_3D, destination);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_R32UI, 1, 1, 3);
    const std::array<GLuint, 3> initialValues = {3u, 5u, 7u};
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, 1, 1, 3, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    initialValues.data());
    ASSERT_GL_NO_ERROR();

    glBindImageTexture(0, source, 0, GL_FALSE, 2, GL_READ_ONLY, GL_R32UI);
    glBindImageTexture(1, destination, 0, GL_FALSE, 1, GL_WRITE_ONLY, GL_R32UI);
    glUseProgram(program);
    glDispatchCompute(1, 1, 1);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT);

    GLFramebuffer framebuffer;
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
    std::array<GLuint, 3> resultValues = {};
    for (GLint layer = 0; layer < 3; ++layer)
    {
        glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, destination, 0, layer);
        ASSERT_GL_FRAMEBUFFER_COMPLETE(GL_READ_FRAMEBUFFER);
        glReadPixels(0, 0, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT,
                     &resultValues[static_cast<size_t>(layer)]);
    }
    EXPECT_GL_NO_ERROR();
    EXPECT_EQ((std::array<GLuint, 3>{3u, 50u, 7u}), resultValues);
}

TEST_P(ExperimentalGles31MetalComputeTest, ImageHelperFunctionParameter3DSliceLoadStore)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x = 1) in;
layout(r32ui, binding = 0) uniform highp uimage2D images[2];
layout(std430, binding = 0) buffer Output
{
    uint loadedValue;
} outputData;

uint loadFromImage(highp uimage2D sourceImage)
{
    return imageLoad(sourceImage, ivec2(0)).r + uint(imageSize(sourceImage).x);
}

void storeToImage(highp uimage2D destinationImage, uint value)
{
    imageStore(destinationImage, ivec2(0), uvec4(value));
}

void main()
{
    uint value = loadFromImage(images[0]);
    outputData.loadedValue = value;
    storeToImage(images[1], value + 1u);
})";
    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);

    GLTexture source;
    GLTexture destination;
    glBindTexture(GL_TEXTURE_3D, source);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_R32UI, 1, 1, 3);
    const std::array<GLuint, 3> sourceValues = {17u, 29u, 41u};
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, 1, 1, 3, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    sourceValues.data());
    glBindTexture(GL_TEXTURE_3D, destination);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_R32UI, 1, 1, 3);
    const std::array<GLuint, 3> destinationValues = {3u, 5u, 7u};
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, 1, 1, 3, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    destinationValues.data());
    GLBuffer output;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, output);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), nullptr, GL_DYNAMIC_DRAW);
    ASSERT_GL_NO_ERROR();

    glBindImageTexture(0, source, 0, GL_FALSE, 2, GL_READ_ONLY, GL_R32UI);
    glBindImageTexture(1, destination, 0, GL_FALSE, 1, GL_WRITE_ONLY, GL_R32UI);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, output);
    glUseProgram(program);
    glDispatchCompute(1, 1, 1);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT |
                    GL_FRAMEBUFFER_BARRIER_BIT);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, output);
    const GLuint *loadedValue = static_cast<const GLuint *>(
        glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, loadedValue);
    const GLuint loaded = *loadedValue;
    EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));

    GLFramebuffer framebuffer;
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
    std::array<GLuint, 3> resultValues = {};
    for (GLint layer = 0; layer < 3; ++layer)
    {
        glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, destination, 0, layer);
        ASSERT_GL_FRAMEBUFFER_COMPLETE(GL_READ_FRAMEBUFFER);
        glReadPixels(0, 0, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT,
                     &resultValues[static_cast<size_t>(layer)]);
    }
    EXPECT_GL_NO_ERROR();
    EXPECT_EQ(42u, loaded);
    EXPECT_EQ((std::array<GLuint, 3>{3u, 43u, 7u}), resultValues);
}

TEST_P(ExperimentalGles31MetalComputeTest, LayeredImageCubeLoadStore)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(r32ui, binding = 0) uniform highp uimageCube image;
void main()
{
    ivec3 coord = ivec3(0, 0, int(gl_GlobalInvocationID.z));
    uint value = imageLoad(image, coord).r;
    imageStore(image, coord, uvec4(value + uint(imageSize(image).x)));
})";
    GLuint computeShader = CompileShader(GL_COMPUTE_SHADER, kCS);
    ASSERT_NE(0u, computeShader);
    ASSERT_GL_NO_ERROR();
    GLuint program = glCreateProgram();
    ASSERT_NE(0u, program);
    glAttachShader(program, computeShader);
    glLinkProgram(program);
    ASSERT_GL_NO_ERROR();
    GLint linkStatus = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
    if (linkStatus != GL_TRUE)
    {
        GLint infoLogLength = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength);
        std::vector<GLchar> infoLog(static_cast<size_t>(std::max(infoLogLength, 1)));
        glGetProgramInfoLog(program, static_cast<GLsizei>(infoLog.size()), nullptr, infoLog.data());
        FAIL() << "Program link failed: " << infoLog.data();
    }

    GLTexture image;
    glBindTexture(GL_TEXTURE_CUBE_MAP, image);
    glTexStorage2D(GL_TEXTURE_CUBE_MAP, 1, GL_R32UI, 1, 1);
    const std::array<GLuint, 6> initialValues = {3u, 5u, 7u, 9u, 11u, 13u};
    for (GLuint face = 0; face < 6; ++face)
    {
        const GLenum faceTarget = GL_TEXTURE_CUBE_MAP_POSITIVE_X + face;
        glTexSubImage2D(faceTarget, 0, 0, 0, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT,
                        &initialValues[face]);
    }
    glBindImageTexture(0, image, 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32UI);

    glUseProgram(program);
    glDispatchCompute(1, 1, 6);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT);

    GLFramebuffer framebuffer;
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
    std::array<GLuint, 6> resultValues = {};
    for (GLuint face = 0; face < 6; ++face)
    {
        const GLenum faceTarget = GL_TEXTURE_CUBE_MAP_POSITIVE_X + face;
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, faceTarget, image, 0);
        ASSERT_GL_FRAMEBUFFER_COMPLETE(GL_READ_FRAMEBUFFER);
        glReadPixels(0, 0, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, &resultValues[face]);
    }
    EXPECT_GL_NO_ERROR();
    EXPECT_EQ((std::array<GLuint, 6>{4u, 6u, 8u, 10u, 12u, 14u}), resultValues);
}

TEST_P(ExperimentalGles31MetalComputeTest, ImageSignedAndFloatLoadStore)
{
    constexpr char kWriteCS[] = R"(#version 310 es
layout(local_size_x = 1) in;
layout(r32i, binding = 0) writeonly uniform highp iimage2D signedImage;
layout(r32f, binding = 1) writeonly uniform highp image2D floatImage;
void main()
{
    imageStore(signedImage, ivec2(0), ivec4(-29));
    imageStore(floatImage, ivec2(0), vec4(3.25));
})";
    constexpr char kReadCS[] = R"(#version 310 es
layout(local_size_x = 1) in;
layout(r32i, binding = 0) readonly uniform highp iimage2D signedImage;
layout(r32f, binding = 1) readonly uniform highp image2D floatImage;
layout(std430, binding = 0) buffer Output
{
    int signedValue;
    float floatValue;
} outputData;
void main()
{
    outputData.signedValue = imageLoad(signedImage, ivec2(0)).r;
    outputData.floatValue = imageLoad(floatImage, ivec2(0)).r;
})";
    ANGLE_GL_COMPUTE_PROGRAM(writeProgram, kWriteCS);
    ANGLE_GL_COMPUTE_PROGRAM(readProgram, kReadCS);

    GLTexture signedImage;
    glBindTexture(GL_TEXTURE_2D, signedImage);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32I, 1, 1);
    GLTexture floatImage;
    glBindTexture(GL_TEXTURE_2D, floatImage);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, 1, 1);
    glBindImageTexture(0, signedImage, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32I);
    glBindImageTexture(1, floatImage, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

    glUseProgram(writeProgram);
    glDispatchCompute(1, 1, 1);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

    GLBuffer outputBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLint) + sizeof(GLfloat), nullptr,
                 GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, outputBuffer);
    glBindImageTexture(0, signedImage, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32I);
    glBindImageTexture(1, floatImage, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
    glUseProgram(readProgram);
    glDispatchCompute(1, 1, 1);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    struct OutputValues
    {
        GLint signedValue;
        GLfloat floatValue;
    };
    const auto *outputValues = static_cast<const OutputValues *>(
        glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeof(OutputValues), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, outputValues);
    EXPECT_EQ(-29, outputValues->signedValue);
    EXPECT_FLOAT_EQ(3.25f, outputValues->floatValue);
    EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));
}

TEST_P(ExperimentalGles31MetalComputeTest, ImageRgbaFormatLoadStore)
{
    constexpr char kWriteCS[] = R"(#version 310 es
layout(local_size_x = 1) in;
layout(rgba32i, binding = 0) writeonly uniform highp iimage2D signedImage;
layout(rgba32ui, binding = 1) writeonly uniform highp uimage2D unsignedImage;
layout(rgba32f, binding = 2) writeonly uniform highp image2D floatImage;
void main()
{
    imageStore(signedImage, ivec2(0), ivec4(-1, -2, -3, -4));
    imageStore(unsignedImage, ivec2(0), uvec4(11u, 12u, 13u, 14u));
    imageStore(floatImage, ivec2(0), vec4(1.25, 2.5, -3.75, 4.5));
})";
    constexpr char kReadCS[] = R"(#version 310 es
layout(local_size_x = 1) in;
layout(rgba32i, binding = 0) readonly uniform highp iimage2D signedImage;
layout(rgba32ui, binding = 1) readonly uniform highp uimage2D unsignedImage;
layout(rgba32f, binding = 2) readonly uniform highp image2D floatImage;
layout(std430, binding = 0) buffer Output
{
    ivec4 signedValues;
    uvec4 unsignedValues;
    vec4 floatValues;
} outputData;
void main()
{
    outputData.signedValues = imageLoad(signedImage, ivec2(0));
    outputData.unsignedValues = imageLoad(unsignedImage, ivec2(0));
    outputData.floatValues = imageLoad(floatImage, ivec2(0));
})";
    ANGLE_GL_COMPUTE_PROGRAM(writeProgram, kWriteCS);
    ANGLE_GL_COMPUTE_PROGRAM(readProgram, kReadCS);

    GLTexture signedImage;
    glBindTexture(GL_TEXTURE_2D, signedImage);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32I, 1, 1);
    GLTexture unsignedImage;
    glBindTexture(GL_TEXTURE_2D, unsignedImage);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32UI, 1, 1);
    GLTexture floatImage;
    glBindTexture(GL_TEXTURE_2D, floatImage);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, 1, 1);
    glBindImageTexture(0, signedImage, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32I);
    glBindImageTexture(1, unsignedImage, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32UI);
    glBindImageTexture(2, floatImage, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

    glUseProgram(writeProgram);
    glDispatchCompute(1, 1, 1);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

    GLBuffer outputBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 3 * sizeof(GLint) * 4, nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, outputBuffer);
    glBindImageTexture(0, signedImage, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32I);
    glBindImageTexture(1, unsignedImage, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32UI);
    glBindImageTexture(2, floatImage, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
    glUseProgram(readProgram);
    glDispatchCompute(1, 1, 1);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    struct OutputValues
    {
        GLint signedValues[4];
        GLuint unsignedValues[4];
        GLfloat floatValues[4];
    };
    const auto *outputValues = static_cast<const OutputValues *>(
        glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeof(OutputValues), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, outputValues);
    EXPECT_EQ(-1, outputValues->signedValues[0]);
    EXPECT_EQ(-2, outputValues->signedValues[1]);
    EXPECT_EQ(-3, outputValues->signedValues[2]);
    EXPECT_EQ(-4, outputValues->signedValues[3]);
    EXPECT_EQ(11u, outputValues->unsignedValues[0]);
    EXPECT_EQ(12u, outputValues->unsignedValues[1]);
    EXPECT_EQ(13u, outputValues->unsignedValues[2]);
    EXPECT_EQ(14u, outputValues->unsignedValues[3]);
    EXPECT_FLOAT_EQ(1.25f, outputValues->floatValues[0]);
    EXPECT_FLOAT_EQ(2.5f, outputValues->floatValues[1]);
    EXPECT_FLOAT_EQ(-3.75f, outputValues->floatValues[2]);
    EXPECT_FLOAT_EQ(4.5f, outputValues->floatValues[3]);
    EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));
}

TEST_P(ExperimentalGles31MetalComputeTest, ImageRemainingCoreFormatsLoadStore)
{
    auto runFormat = [&](const char *formatQualifier, GLenum internalFormat, const char *imageType,
                         const char *vectorType, const char *components, auto expected,
                         auto expectValue) {
        using ComponentType = typename decltype(expected)::value_type;
        SCOPED_TRACE(formatQualifier);

        std::string writeCS = "#version 310 es\nlayout(local_size_x = 1) in;\n";
        writeCS += "layout(" + std::string(formatQualifier) +
                   ", binding = 0) writeonly uniform highp " + imageType + " image;\n";
        writeCS += std::string("void main() { imageStore(image, ivec2(0), ") + vectorType + "(" +
                   components + ")); }\n";

        std::string readCS = "#version 310 es\nlayout(local_size_x = 1) in;\n";
        readCS += "layout(" + std::string(formatQualifier) +
                  ", binding = 0) readonly uniform highp " + imageType + " image;\n";
        readCS += "layout(std430, binding = 0) buffer Output { " + std::string(vectorType) +
                  " values; } outputData;\n";
        readCS += "void main() { outputData.values = imageLoad(image, ivec2(0)); }\n";

        GLProgram writeProgram;
        writeProgram.makeCompute(writeCS.c_str());
        ASSERT_TRUE(writeProgram.valid());
        GLProgram readProgram;
        readProgram.makeCompute(readCS.c_str());
        ASSERT_TRUE(readProgram.valid());

        GLTexture image;
        glBindTexture(GL_TEXTURE_2D, image);
        glTexStorage2D(GL_TEXTURE_2D, 1, internalFormat, 1, 1);
        glBindImageTexture(0, image, 0, GL_FALSE, 0, GL_WRITE_ONLY, internalFormat);
        GLBuffer outputBuffer;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(expected), nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, outputBuffer);

        glUseProgram(writeProgram);
        glDispatchCompute(1, 1, 1);
        ASSERT_GL_NO_ERROR();
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

        glBindImageTexture(0, image, 0, GL_FALSE, 0, GL_READ_ONLY, internalFormat);
        glUseProgram(readProgram);
        glDispatchCompute(1, 1, 1);
        ASSERT_GL_NO_ERROR();
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

        const auto *values = static_cast<const ComponentType *>(glMapBufferRange(
            GL_SHADER_STORAGE_BUFFER, 0, sizeof(expected), GL_MAP_READ_BIT));
        ASSERT_NE(nullptr, values);
        for (size_t component = 0; component < expected.size(); ++component)
        {
            expectValue(expected[component], values[component]);
        }
        EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));
    };

    runFormat("rgba16f", GL_RGBA16F, "image2D", "vec4", "1.5, -2.0, 0.5, 4.0",
              std::array<GLfloat, 4>{1.5f, -2.0f, 0.5f, 4.0f},
              [](GLfloat expected, GLfloat actual) { EXPECT_FLOAT_EQ(expected, actual); });
    runFormat("rgba8", GL_RGBA8, "image2D", "vec4", "0.0, 1.0, 2.0, -1.0",
              std::array<GLfloat, 4>{0.0f, 1.0f, 1.0f, 0.0f},
              [](GLfloat expected, GLfloat actual) { EXPECT_FLOAT_EQ(expected, actual); });
    runFormat("rgba8_snorm", GL_RGBA8_SNORM, "image2D", "vec4", "-1.0, 0.0, 0.0, 1.0",
              std::array<GLfloat, 4>{-1.0f, 0.0f, 0.0f, 1.0f},
              [](GLfloat expected, GLfloat actual) { EXPECT_FLOAT_EQ(expected, actual); });
    runFormat("rgba16i", GL_RGBA16I, "iimage2D", "ivec4", "-1, 0, 1, 32767",
              std::array<GLint, 4>{-1, 0, 1, 32767},
              [](GLint expected, GLint actual) { EXPECT_EQ(expected, actual); });
    runFormat("rgba8i", GL_RGBA8I, "iimage2D", "ivec4", "-128, -1, 0, 127",
              std::array<GLint, 4>{-128, -1, 0, 127},
              [](GLint expected, GLint actual) { EXPECT_EQ(expected, actual); });
    runFormat("rgba16ui", GL_RGBA16UI, "uimage2D", "uvec4", "0u, 1u, 32768u, 65535u",
              std::array<GLuint, 4>{0u, 1u, 32768u, 65535u},
              [](GLuint expected, GLuint actual) { EXPECT_EQ(expected, actual); });
    runFormat("rgba8ui", GL_RGBA8UI, "uimage2D", "uvec4", "0u, 1u, 127u, 255u",
              std::array<GLuint, 4>{0u, 1u, 127u, 255u},
              [](GLuint expected, GLuint actual) { EXPECT_EQ(expected, actual); });
}

TEST_P(ExperimentalGles31MetalComputeTest, FragmentStageImageLoad)
{
    constexpr char kVS[] = R"(#version 310 es
layout(location = 0) in vec2 position;
void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
})";
    constexpr char kFS[] = R"(#version 310 es
precision highp float;
layout(r32ui, binding = 0) readonly uniform highp uimage2D sourceImage;
layout(location = 0) out vec4 fragColor;
void main()
{
    fragColor = vec4(float(imageLoad(sourceImage, ivec2(0)).r), 0.0, 0.0, 1.0);
})";
    ANGLE_GL_PROGRAM(program, kVS, kFS);

    GLTexture image;
    glBindTexture(GL_TEXTURE_2D, image);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, 1, 1);
    const GLuint initialValue = 1;
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    &initialValue);
    glBindImageTexture(0, image, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI);
    ASSERT_GL_NO_ERROR();

    drawQuad(program, "position", 0.0f, 1.0f);
    EXPECT_GL_NO_ERROR();
    EXPECT_PIXEL_COLOR_EQ(getWindowWidth() / 2, getWindowHeight() / 2, GLColor::red);
}

TEST_P(ExperimentalGles31MetalComputeTest, FragmentStageImageStore)
{
    constexpr char kVS[] = R"(#version 310 es
layout(location = 0) in vec2 position;
void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
})";
    constexpr char kFS[] = R"(#version 310 es
precision highp float;
layout(r32ui, binding = 0) writeonly uniform highp uimage2D destinationImage;
layout(location = 0) out vec4 fragColor;
void main()
{
    imageStore(destinationImage, ivec2(gl_FragCoord.xy), uvec4(37u));
    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
})";
    ANGLE_GL_PROGRAM(program, kVS, kFS);

    GLTexture image;
    glBindTexture(GL_TEXTURE_2D, image);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, getWindowWidth(), getWindowHeight());
    glBindImageTexture(0, image, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
    ASSERT_GL_NO_ERROR();

    drawQuad(program, "position", 0.0f, 1.0f);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT);

    GLFramebuffer framebuffer;
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, image, 0);
    ASSERT_GL_FRAMEBUFFER_COMPLETE(GL_READ_FRAMEBUFFER);
    GLuint value = 0;
    glReadPixels(getWindowWidth() / 2, getWindowHeight() / 2, 1, 1, GL_RED_INTEGER,
                 GL_UNSIGNED_INT, &value);
    EXPECT_GL_NO_ERROR();
    EXPECT_EQ(37u, value);
}

TEST_P(ExperimentalGles31MetalComputeTest, VertexStageImageStore)
{
    constexpr char kVS[] = R"(#version 310 es
layout(location = 0) in vec2 position;
layout(r32ui, binding = 0) writeonly uniform highp uimage2D destinationImage;
void main()
{
    imageStore(destinationImage, ivec2(gl_VertexID, 0), uvec4(uint(gl_VertexID) + 10u));
    gl_Position = vec4(position, 0.0, 1.0);
})";
    constexpr char kFS[] = R"(#version 310 es
precision highp float;
layout(location = 0) out vec4 fragColor;
void main()
{
    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
})";
    ANGLE_GL_PROGRAM(program, kVS, kFS);

    GLTexture image;
    glBindTexture(GL_TEXTURE_2D, image);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, 3, 1);
    glBindImageTexture(0, image, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);

    constexpr std::array<GLfloat, 6> positions = {-0.5f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f};
    GLBuffer vertexBuffer;
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);

    glUseProgram(program);
    glDrawArrays(GL_POINTS, 0, 3);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT);

    GLFramebuffer framebuffer;
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, image, 0);
    ASSERT_GL_FRAMEBUFFER_COMPLETE(GL_READ_FRAMEBUFFER);
    std::array<GLuint, 3> values = {};
    glReadPixels(0, 0, 3, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, values.data());
    EXPECT_GL_NO_ERROR();
    EXPECT_EQ((std::array<GLuint, 3>{10u, 11u, 12u}), values);
}

TEST_P(ExperimentalGles31MetalComputeTest, GraphicsStageLayeredImageStores)
{
    constexpr char kLayeredVS[] = R"(#version 310 es
layout(r32ui, binding = 0) writeonly uniform highp uimage2DArray arrayImage;
layout(r32ui, binding = 1) writeonly uniform highp uimage3D volumeImage;
layout(r32ui, binding = 2) writeonly uniform highp uimageCube cubeImage;
void main()
{
    imageStore(arrayImage, ivec3(0, 0, 0), uvec4(11u));
    imageStore(volumeImage, ivec3(0, 0, 0), uvec4(21u));
    imageStore(cubeImage, ivec3(0, 0, 0), uvec4(31u));
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
    gl_PointSize = 1.0;
})";
    constexpr char kLayeredFS[] = R"(#version 310 es
precision highp float;
layout(r32ui, binding = 0) writeonly uniform highp uimage2DArray arrayImage;
layout(r32ui, binding = 1) writeonly uniform highp uimage3D volumeImage;
layout(r32ui, binding = 2) writeonly uniform highp uimageCube cubeImage;
out vec4 color;
void main()
{
    imageStore(arrayImage, ivec3(0, 0, 1), uvec4(12u));
    imageStore(volumeImage, ivec3(0, 0, 1), uvec4(22u));
    imageStore(cubeImage, ivec3(0, 0, 4), uvec4(32u));
    color = vec4(0.0, 0.0, 0.0, 1.0);
})";
    constexpr char kPlainVS[]   = R"(#version 310 es
void main()
{
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
    gl_PointSize = 1.0;
})";
    constexpr char kPlainFS[]   = R"(#version 310 es
precision highp float;
out vec4 color;
void main()
{
    color = vec4(0.0, 0.0, 0.0, 1.0);
})";
    ANGLE_GL_PROGRAM(vertexProgram, kLayeredVS, kPlainFS);
    ANGLE_GL_PROGRAM(fragmentProgram, kPlainVS, kLayeredFS);

    GLTexture arrayImage;
    glBindTexture(GL_TEXTURE_2D_ARRAY, arrayImage);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_R32UI, 1, 1, 2);
    GLTexture volumeImage;
    glBindTexture(GL_TEXTURE_3D, volumeImage);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_R32UI, 1, 1, 2);
    GLTexture cubeImage;
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubeImage);
    glTexStorage2D(GL_TEXTURE_CUBE_MAP, 1, GL_R32UI, 1, 1);
    ASSERT_GL_NO_ERROR();

    glBindImageTexture(0, arrayImage, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32UI);
    glBindImageTexture(1, volumeImage, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32UI);
    glBindImageTexture(2, cubeImage, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32UI);
    GLTexture framebufferTexture;
    glBindTexture(GL_TEXTURE_2D, framebufferTexture);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 1, 1);
    GLFramebuffer framebuffer;
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, framebufferTexture,
                           0);
    ASSERT_GL_FRAMEBUFFER_COMPLETE(GL_FRAMEBUFFER);
    glViewport(0, 0, 1, 1);

    glUseProgram(vertexProgram);
    glDrawArrays(GL_POINTS, 0, 1);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT);
    glUseProgram(fragmentProgram);
    glDrawArrays(GL_POINTS, 0, 1);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT);

    std::array<GLuint, 6> values = {};
    glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, arrayImage, 0, 0);
    ASSERT_GL_FRAMEBUFFER_COMPLETE(GL_READ_FRAMEBUFFER);
    glReadPixels(0, 0, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, &values[0]);
    glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, arrayImage, 0, 1);
    ASSERT_GL_FRAMEBUFFER_COMPLETE(GL_READ_FRAMEBUFFER);
    glReadPixels(0, 0, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, &values[1]);
    glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, volumeImage, 0, 0);
    ASSERT_GL_FRAMEBUFFER_COMPLETE(GL_READ_FRAMEBUFFER);
    glReadPixels(0, 0, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, &values[2]);
    glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, volumeImage, 0, 1);
    ASSERT_GL_FRAMEBUFFER_COMPLETE(GL_READ_FRAMEBUFFER);
    glReadPixels(0, 0, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, &values[3]);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_CUBE_MAP_POSITIVE_X, cubeImage, 0);
    ASSERT_GL_FRAMEBUFFER_COMPLETE(GL_READ_FRAMEBUFFER);
    glReadPixels(0, 0, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, &values[4]);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_CUBE_MAP_POSITIVE_X + 4, cubeImage, 0);
    ASSERT_GL_FRAMEBUFFER_COMPLETE(GL_READ_FRAMEBUFFER);
    glReadPixels(0, 0, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, &values[5]);
    EXPECT_GL_NO_ERROR();
    EXPECT_EQ((std::array<GLuint, 6>{11u, 12u, 21u, 22u, 31u, 32u}), values);
}

TEST_P(ExperimentalGles31MetalComputeTest, GraphicsStageNonLayered3DImage2DStores)
{
    constexpr char kImageVS[] = R"(#version 310 es
layout(r32ui, binding = 0) writeonly uniform highp uimage2D image;
void main()
{
    imageStore(image, ivec2(0), uvec4(41u));
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
    gl_PointSize = 1.0;
})";
    constexpr char kImageFS[] = R"(#version 310 es
precision highp float;
layout(r32ui, binding = 0) writeonly uniform highp uimage2D image;
out vec4 color;
void main()
{
    imageStore(image, ivec2(0), uvec4(42u));
    color = vec4(0.0, 0.0, 0.0, 1.0);
})";
    constexpr char kPlainVS[] = R"(#version 310 es
void main()
{
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
    gl_PointSize = 1.0;
})";
    constexpr char kPlainFS[] = R"(#version 310 es
precision highp float;
out vec4 color;
void main()
{
    color = vec4(0.0, 0.0, 0.0, 1.0);
})";
    ANGLE_GL_PROGRAM(vertexProgram, kImageVS, kPlainFS);
    ANGLE_GL_PROGRAM(fragmentProgram, kPlainVS, kImageFS);

    GLTexture image3D;
    glBindTexture(GL_TEXTURE_3D, image3D);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_R32UI, 1, 1, 2);
    ASSERT_GL_NO_ERROR();

    GLTexture framebufferTexture;
    glBindTexture(GL_TEXTURE_2D, framebufferTexture);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 1, 1);
    GLFramebuffer framebuffer;
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, framebufferTexture,
                           0);
    ASSERT_GL_FRAMEBUFFER_COMPLETE(GL_FRAMEBUFFER);
    glViewport(0, 0, 1, 1);

    glBindImageTexture(0, image3D, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
    glUseProgram(vertexProgram);
    glDrawArrays(GL_POINTS, 0, 1);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT);

    glBindImageTexture(0, image3D, 0, GL_FALSE, 1, GL_WRITE_ONLY, GL_R32UI);
    glUseProgram(fragmentProgram);
    glDrawArrays(GL_POINTS, 0, 1);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT);

    std::array<GLuint, 2> values = {};
    glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, image3D, 0, 0);
    ASSERT_GL_FRAMEBUFFER_COMPLETE(GL_READ_FRAMEBUFFER);
    glReadPixels(0, 0, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, &values[0]);
    glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, image3D, 0, 1);
    ASSERT_GL_FRAMEBUFFER_COMPLETE(GL_READ_FRAMEBUFFER);
    glReadPixels(0, 0, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, &values[1]);
    EXPECT_GL_NO_ERROR();
    EXPECT_EQ((std::array<GLuint, 2>{41u, 42u}), values);
}

TEST_P(ExperimentalGles31MetalComputeTest, GraphicsStageImageFormatsLoadStore)
{
    auto runFormat = [&](const char *formatQualifier, GLenum internalFormat, const char *imageType,
                         const char *vectorType, const char *components, auto expected,
                         auto expectValue) {
        using ComponentType = typename decltype(expected)::value_type;
        SCOPED_TRACE(formatQualifier);

        const std::string declaration = "layout(" + std::string(formatQualifier) + ", binding = ";
        std::string initializeCS      = R"(#version 310 es
layout(local_size_x = 1) in;
)";
        initializeCS += declaration + "0) writeonly uniform highp " + imageType + " image;\n";
        initializeCS += std::string("void main() { imageStore(image, ivec2(0), ") + vectorType +
                        "(" + components + ")); }\n";

        std::string readCS = R"(#version 310 es
layout(local_size_x = 1) in;
)";
        readCS += declaration + "0) readonly uniform highp " + imageType + " image;\n";
        readCS += "layout(std430, binding = 0) buffer Output { " + std::string(vectorType) +
                  " value; } outputData;\n";
        readCS += "void main() { outputData.value = imageLoad(image, ivec2(0)); }\n";

        GLProgram initializeProgram;
        initializeProgram.makeCompute(initializeCS.c_str());
        ASSERT_TRUE(initializeProgram.valid());
        GLProgram readProgram;
        readProgram.makeCompute(readCS.c_str());
        ASSERT_TRUE(readProgram.valid());

        const std::string sourceDeclaration =
            declaration + "0) readonly uniform highp " + imageType + " sourceImage;\n";
        const std::string destinationDeclaration =
            declaration + "1) writeonly uniform highp " + imageType + " destinationImage;\n";
        const std::string copyImage =
            "imageStore(destinationImage, ivec2(0), imageLoad(sourceImage, ivec2(0)));\n";
        const std::string vertexShader = std::string("#version 310 es\n") + sourceDeclaration +
                                         destinationDeclaration + "void main() { " + copyImage +
                                         "gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n";
        constexpr char kFragmentOnlyVertexShader[]  = R"(#version 310 es
void main()
{
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
})";
        constexpr char kVertexStageFragmentShader[] = R"(#version 310 es
precision highp float;
out vec4 fragColor;
void main()
{
    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
})";
        const std::string fragmentShader =
            std::string("#version 310 es\nprecision highp float;\n") + sourceDeclaration +
            destinationDeclaration + "out vec4 fragColor;\nvoid main() { " + copyImage +
            "fragColor = vec4(0.0, 0.0, 0.0, 1.0); }\n";
        GLProgram vertexProgram;
        vertexProgram.makeRaster(vertexShader.c_str(), kVertexStageFragmentShader);
        ASSERT_TRUE(vertexProgram.valid());
        GLProgram fragmentProgram;
        fragmentProgram.makeRaster(kFragmentOnlyVertexShader, fragmentShader.c_str());
        ASSERT_TRUE(fragmentProgram.valid());

        GLTexture sourceImage;
        glBindTexture(GL_TEXTURE_2D, sourceImage);
        glTexStorage2D(GL_TEXTURE_2D, 1, internalFormat, 1, 1);
        GLTexture destinationImage;
        glBindTexture(GL_TEXTURE_2D, destinationImage);
        glTexStorage2D(GL_TEXTURE_2D, 1, internalFormat, 1, 1);
        glBindImageTexture(0, sourceImage, 0, GL_FALSE, 0, GL_WRITE_ONLY, internalFormat);
        glUseProgram(initializeProgram);
        glDispatchCompute(1, 1, 1);
        ASSERT_GL_NO_ERROR();
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        GLBuffer outputBuffer;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(expected), nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, outputBuffer);

        const std::array<GLProgram *, 2> graphicsPrograms = {&vertexProgram, &fragmentProgram};
        for (size_t stage = 0; stage < graphicsPrograms.size(); ++stage)
        {
            glBindImageTexture(0, sourceImage, 0, GL_FALSE, 0, GL_READ_ONLY, internalFormat);
            glBindImageTexture(1, destinationImage, 0, GL_FALSE, 0, GL_WRITE_ONLY, internalFormat);
            glUseProgram(*graphicsPrograms[stage]);
            glDrawArrays(GL_POINTS, 0, 1);
            ASSERT_GL_NO_ERROR();
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

            glBindImageTexture(0, destinationImage, 0, GL_FALSE, 0, GL_READ_ONLY, internalFormat);
            glUseProgram(readProgram);
            glDispatchCompute(1, 1, 1);
            ASSERT_GL_NO_ERROR();
            glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

            const auto *values = static_cast<const ComponentType *>(glMapBufferRange(
                GL_SHADER_STORAGE_BUFFER, 0, sizeof(expected), GL_MAP_READ_BIT));
            ASSERT_NE(nullptr, values);
            for (size_t component = 0; component < expected.size(); ++component)
            {
                SCOPED_TRACE(stage == 0 ? "vertex stage" : "fragment stage");
                expectValue(expected[component], values[component]);
            }
            EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));
        }
        EXPECT_GL_NO_ERROR();
    };

    runFormat("r32f", GL_R32F, "image2D", "vec4", "3.25, 0.0, 0.0, 1.0",
              std::array<GLfloat, 4>{3.25f, 0.0f, 0.0f, 1.0f},
              [](GLfloat expected, GLfloat actual) { EXPECT_FLOAT_EQ(expected, actual); });
    runFormat("r32i", GL_R32I, "iimage2D", "ivec4", "-29, 0, 0, 1",
              std::array<GLint, 4>{-29, 0, 0, 1},
              [](GLint expected, GLint actual) { EXPECT_EQ(expected, actual); });
    runFormat("r32ui", GL_R32UI, "uimage2D", "uvec4", "17u, 0u, 0u, 1u",
              std::array<GLuint, 4>{17u, 0u, 0u, 1u},
              [](GLuint expected, GLuint actual) { EXPECT_EQ(expected, actual); });
    runFormat("rgba32f", GL_RGBA32F, "image2D", "vec4", "1.25, 2.5, -3.75, 4.5",
              std::array<GLfloat, 4>{1.25f, 2.5f, -3.75f, 4.5f},
              [](GLfloat expected, GLfloat actual) { EXPECT_FLOAT_EQ(expected, actual); });
    runFormat("rgba32i", GL_RGBA32I, "iimage2D", "ivec4", "-1, -2, -3, -4",
              std::array<GLint, 4>{-1, -2, -3, -4},
              [](GLint expected, GLint actual) { EXPECT_EQ(expected, actual); });
    runFormat("rgba32ui", GL_RGBA32UI, "uimage2D", "uvec4", "11u, 12u, 13u, 14u",
              std::array<GLuint, 4>{11u, 12u, 13u, 14u},
              [](GLuint expected, GLuint actual) { EXPECT_EQ(expected, actual); });
    runFormat("rgba16f", GL_RGBA16F, "image2D", "vec4", "1.5, -2.0, 0.5, 4.0",
              std::array<GLfloat, 4>{1.5f, -2.0f, 0.5f, 4.0f},
              [](GLfloat expected, GLfloat actual) { EXPECT_FLOAT_EQ(expected, actual); });
    runFormat("rgba8", GL_RGBA8, "image2D", "vec4", "0.0, 1.0, 0.0, 1.0",
              std::array<GLfloat, 4>{0.0f, 1.0f, 0.0f, 1.0f},
              [](GLfloat expected, GLfloat actual) { EXPECT_FLOAT_EQ(expected, actual); });
    runFormat("rgba8_snorm", GL_RGBA8_SNORM, "image2D", "vec4", "-1.0, 0.0, 0.0, 1.0",
              std::array<GLfloat, 4>{-1.0f, 0.0f, 0.0f, 1.0f},
              [](GLfloat expected, GLfloat actual) { EXPECT_FLOAT_EQ(expected, actual); });
    runFormat("rgba16i", GL_RGBA16I, "iimage2D", "ivec4", "-1, 0, 1, 32767",
              std::array<GLint, 4>{-1, 0, 1, 32767},
              [](GLint expected, GLint actual) { EXPECT_EQ(expected, actual); });
    runFormat("rgba8i", GL_RGBA8I, "iimage2D", "ivec4", "-128, -1, 0, 127",
              std::array<GLint, 4>{-128, -1, 0, 127},
              [](GLint expected, GLint actual) { EXPECT_EQ(expected, actual); });
    runFormat("rgba16ui", GL_RGBA16UI, "uimage2D", "uvec4", "0u, 1u, 32768u, 65535u",
              std::array<GLuint, 4>{0u, 1u, 32768u, 65535u},
              [](GLuint expected, GLuint actual) { EXPECT_EQ(expected, actual); });
    runFormat("rgba8ui", GL_RGBA8UI, "uimage2D", "uvec4", "0u, 1u, 127u, 255u",
              std::array<GLuint, 4>{0u, 1u, 127u, 255u},
              [](GLuint expected, GLuint actual) { EXPECT_EQ(expected, actual); });
}

TEST_P(ExperimentalGles31MetalComputeTest, GraphicsStageStorageBufferAccess)
{
    constexpr char kVS[] = R"(#version 310 es
layout(location = 0) in vec2 position;
layout(std430, binding = 0) readonly buffer VertexData
{
    float value;
} vertexData;
layout(std430, binding = 2) writeonly buffer VertexOutput
{
    float values[];
} vertexOutput;
out float vertexValue;
void main()
{
    vertexValue = vertexData.value;
    vertexOutput.values[gl_VertexID] = vertexData.value;
    gl_Position = vec4(position, 0.0, 1.0);
})";
    constexpr char kFS[] = R"(#version 310 es
precision highp float;
layout(std430, binding = 1) readonly buffer FragmentData
{
    vec3 color;
} fragmentData;
in float vertexValue;
layout(location = 0) out vec4 fragColor;
void main()
{
    fragColor = vec4(fragmentData.color.r, vertexValue, fragmentData.color.b, 1.0);
})";

    ANGLE_GL_PROGRAM(program, kVS, kFS);

    const GLfloat vertexValue = 0.0f;
    const std::array<GLfloat, 3> fragmentColor = {1.0f, 0.0f, 1.0f};
    GLBuffer vertexBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(vertexValue), &vertexValue, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vertexBuffer);

    GLBuffer fragmentBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, fragmentBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(fragmentColor), fragmentColor.data(),
                 GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, fragmentBuffer);

    std::array<GLfloat, 6> outputValues = {};
    GLBuffer outputBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(outputValues), outputValues.data(),
                 GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, outputBuffer);

    drawQuad(program, "position", 0.0f, 1.0f);
    ASSERT_GL_NO_ERROR();
    EXPECT_PIXEL_COLOR_EQ(getWindowWidth() / 2, getWindowHeight() / 2, GLColor::magenta);

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    const auto *writtenValues = static_cast<const GLfloat *>(glMapBufferRange(
        GL_SHADER_STORAGE_BUFFER, 0, sizeof(outputValues), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, writtenValues);
    for (size_t i = 0; i < outputValues.size(); ++i)
    {
        EXPECT_EQ(0.0f, writtenValues[i]);
    }
    EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));
    EXPECT_GL_NO_ERROR();
}

TEST_P(ExperimentalGles31MetalComputeTest, ShaderStorageBufferBlockArray)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Data
{
    uint lengthResult;
    uint values[];
} data[2];
void main()
{
    uint firstLength = uint(data[0].values.length());
    uint secondLength = uint(data[1].values.length());
    data[0].lengthResult = firstLength;
    data[1].lengthResult = secondLength;
    data[0].values[0] = firstLength * 100u;
    data[1].values[0] = secondLength * 100u;
})";
    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);

    std::array<GLuint, 4> firstData = {0, 10, 20, 30};
    GLBuffer firstBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, firstBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(firstData), firstData.data(), GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, firstBuffer);

    std::array<GLuint, 3> secondData = {0, 40, 50};
    GLBuffer secondBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, secondBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(secondData), secondData.data(), GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, secondBuffer);

    glUseProgram(program);
    glDispatchCompute(1, 1, 1);
    ASSERT_GL_NO_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, firstBuffer);
    const auto *firstResult = static_cast<const GLuint *>(
        glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeof(firstData), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, firstResult);
    EXPECT_EQ(3u, firstResult[0]);
    EXPECT_EQ(300u, firstResult[1]);
    EXPECT_EQ(20u, firstResult[2]);
    EXPECT_EQ(30u, firstResult[3]);
    EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, secondBuffer);
    const auto *secondResult = static_cast<const GLuint *>(
        glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeof(secondData), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, secondResult);
    EXPECT_EQ(2u, secondResult[0]);
    EXPECT_EQ(200u, secondResult[1]);
    EXPECT_EQ(50u, secondResult[2]);
    EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));
    EXPECT_GL_NO_ERROR();
}

TEST_P(ExperimentalGles31MetalComputeTest, GraphicsStageShaderStorageBufferBlockArrays)
{
    constexpr char kVertexArrayVS[] = R"(#version 310 es
layout(location = 0) in vec2 position;
layout(std430, binding = 0) readonly buffer VertexData
{
    float value;
} vertexData[2];
out float vertexValue;
void main()
{
    vertexValue = vertexData[0].value + vertexData[1].value;
    gl_Position = vec4(position, 0.0, 1.0);
})";
    constexpr char kVertexArrayFS[] = R"(#version 310 es
precision highp float;
in float vertexValue;
layout(location = 0) out vec4 fragColor;
void main()
{
    fragColor = vec4(vertexValue, 0.0, 0.0, 1.0);
})";
    ANGLE_GL_PROGRAM(vertexArrayProgram, kVertexArrayVS, kVertexArrayFS);

    const GLfloat vertexValues[2] = {0.25f, 0.75f};
    GLBuffer vertexBuffer0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexBuffer0);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLfloat), &vertexValues[0], GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vertexBuffer0);
    GLBuffer vertexBuffer1;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexBuffer1);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLfloat), &vertexValues[1], GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vertexBuffer1);

    drawQuad(vertexArrayProgram, "position", 0.0f, 1.0f);
    EXPECT_GL_NO_ERROR();
    EXPECT_PIXEL_COLOR_EQ(getWindowWidth() / 2, getWindowHeight() / 2, GLColor::red);

    constexpr char kFragmentArrayVS[] = R"(#version 310 es
layout(location = 0) in vec2 position;
void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
})";
    constexpr char kFragmentArrayFS[] = R"(#version 310 es
precision highp float;
layout(std430, binding = 0) readonly buffer FragmentData
{
    vec3 color;
} fragmentData[2];
layout(location = 0) out vec4 fragColor;
void main()
{
    fragColor = vec4(fragmentData[0].color.r, fragmentData[1].color.g, 0.0, 1.0);
})";
    ANGLE_GL_PROGRAM(fragmentArrayProgram, kFragmentArrayVS, kFragmentArrayFS);

    const std::array<GLfloat, 3> fragmentColor0 = {1.0f, 0.0f, 0.0f};
    const std::array<GLfloat, 3> fragmentColor1 = {0.0f, 1.0f, 0.0f};
    GLBuffer fragmentBuffer0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, fragmentBuffer0);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(fragmentColor0), fragmentColor0.data(),
                 GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, fragmentBuffer0);
    GLBuffer fragmentBuffer1;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, fragmentBuffer1);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(fragmentColor1), fragmentColor1.data(),
                 GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, fragmentBuffer1);

    drawQuad(fragmentArrayProgram, "position", 0.0f, 1.0f);
    EXPECT_GL_NO_ERROR();
    EXPECT_PIXEL_COLOR_EQ(getWindowWidth() / 2, getWindowHeight() / 2, GLColor::yellow);
}

// link a simple compute program. It should be successful.
TEST_P(ComputeShaderTest, LinkComputeProgram)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=1) in;
void main()
{
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);

    EXPECT_GL_NO_ERROR();
}

// Link a simple compute program. Then detach the shader and dispatch compute.
// It should be successful.
TEST_P(ComputeShaderTest, DetachShaderAfterLinkSuccess)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=1) in;
void main()
{
})";

    GLuint program = glCreateProgram();

    GLuint cs = CompileShader(GL_COMPUTE_SHADER, kCS);
    EXPECT_NE(0u, cs);

    glAttachShader(program, cs);
    glDeleteShader(cs);

    glLinkProgram(program);
    GLint linkStatus;
    glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
    EXPECT_GL_TRUE(linkStatus);

    glDetachShader(program, cs);
    EXPECT_GL_NO_ERROR();

    glUseProgram(program);
    glDispatchCompute(8, 4, 2);
    EXPECT_GL_NO_ERROR();
}

// link a simple compute program. There is no local size and linking should fail.
TEST_P(ComputeShaderTest, LinkComputeProgramNoLocalSizeLinkError)
{
    constexpr char kCS[] = R"(#version 310 es
void main()
{
})";

    GLuint program = CompileComputeProgram(kCS, false);
    EXPECT_EQ(0u, program);

    glDeleteProgram(program);

    EXPECT_GL_NO_ERROR();
}

// link a simple compute program.
// make sure that uniforms and uniform samplers get recorded
TEST_P(ComputeShaderTest, LinkComputeProgramWithUniforms)
{
    constexpr char kCS[] = R"(#version 310 es
precision mediump sampler2D;
layout(local_size_x=1) in;
uniform int myUniformInt;
uniform sampler2D myUniformSampler;
layout(rgba32i) uniform highp writeonly iimage2D imageOut;
void main()
{
    int q = myUniformInt;
    vec4 v = textureLod(myUniformSampler, vec2(0.0), 0.0);
    imageStore(imageOut, ivec2(0), ivec4(v) * q);
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);

    GLint uniformLoc = glGetUniformLocation(program, "myUniformInt");
    EXPECT_NE(-1, uniformLoc);

    uniformLoc = glGetUniformLocation(program, "myUniformSampler");
    EXPECT_NE(-1, uniformLoc);

    EXPECT_GL_NO_ERROR();
}

// Attach both compute and non-compute shaders. A link time error should occur.
// OpenGL ES 3.10, 7.3 Program Objects
TEST_P(ComputeShaderTest, AttachMultipleShaders)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=1) in;
void main()
{
})";

    constexpr char kVS[] = R"(#version 310 es
void main()
{
})";

    constexpr char kFS[] = R"(#version 310 es
void main()
{
})";

    GLuint program = glCreateProgram();

    GLuint vs = CompileShader(GL_VERTEX_SHADER, kVS);
    GLuint fs = CompileShader(GL_FRAGMENT_SHADER, kFS);
    GLuint cs = CompileShader(GL_COMPUTE_SHADER, kCS);

    EXPECT_NE(0u, vs);
    EXPECT_NE(0u, fs);
    EXPECT_NE(0u, cs);

    glAttachShader(program, vs);
    glDeleteShader(vs);

    glAttachShader(program, fs);
    glDeleteShader(fs);

    glAttachShader(program, cs);
    glDeleteShader(cs);

    glLinkProgram(program);

    GLint linkStatus;
    glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);

    EXPECT_GL_FALSE(linkStatus);

    EXPECT_GL_NO_ERROR();
}

// Attach a vertex, fragment and compute shader.
// Query for the number of attached shaders and check the count.
TEST_P(ComputeShaderTest, AttachmentCount)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=1) in;
void main()
{
})";

    constexpr char kVS[] = R"(#version 310 es
void main()
{
})";

    constexpr char kFS[] = R"(#version 310 es
void main()
{
})";

    GLuint program = glCreateProgram();

    GLuint vs = CompileShader(GL_VERTEX_SHADER, kVS);
    GLuint fs = CompileShader(GL_FRAGMENT_SHADER, kFS);
    GLuint cs = CompileShader(GL_COMPUTE_SHADER, kCS);

    EXPECT_NE(0u, vs);
    EXPECT_NE(0u, fs);
    EXPECT_NE(0u, cs);

    glAttachShader(program, vs);
    glDeleteShader(vs);

    glAttachShader(program, fs);
    glDeleteShader(fs);

    glAttachShader(program, cs);
    glDeleteShader(cs);

    GLint numAttachedShaders;
    glGetProgramiv(program, GL_ATTACHED_SHADERS, &numAttachedShaders);

    EXPECT_EQ(3, numAttachedShaders);

    glDeleteProgram(program);

    EXPECT_GL_NO_ERROR();
}

// Attach a compute shader and link, but start rendering.
TEST_P(ComputeShaderTest, StartRenderingWithComputeProgram)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=1) in;
void main()
{
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    EXPECT_GL_NO_ERROR();

    glUseProgram(program);
    glDrawArrays(GL_POINTS, 0, 2);
    EXPECT_GL_NO_ERROR();
}

// Attach a vertex and fragment shader and link, but dispatch compute.
TEST_P(ComputeShaderTest, DispatchComputeWithRenderingProgram)
{
    constexpr char kVS[] = R"(#version 310 es
void main() {})";

    constexpr char kFS[] = R"(#version 310 es
void main() {})";

    GLuint program = glCreateProgram();

    GLuint vs = CompileShader(GL_VERTEX_SHADER, kVS);
    GLuint fs = CompileShader(GL_FRAGMENT_SHADER, kFS);

    EXPECT_NE(0u, vs);
    EXPECT_NE(0u, fs);

    glAttachShader(program, vs);
    glDeleteShader(vs);

    glAttachShader(program, fs);
    glDeleteShader(fs);

    glLinkProgram(program);

    GLint linkStatus;
    glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
    EXPECT_GL_TRUE(linkStatus);

    EXPECT_GL_NO_ERROR();

    glUseProgram(program);
    glDispatchCompute(8, 4, 2);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
}

// Access all compute shader special variables.
TEST_P(ComputeShaderTest, AccessAllSpecialVariables)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=4, local_size_y=3, local_size_z=2) in;
layout(rgba32ui) uniform highp writeonly uimage2D imageOut;
void main()
{
    uvec3 temp1 = gl_NumWorkGroups;
    uvec3 temp2 = gl_WorkGroupSize;
    uvec3 temp3 = gl_WorkGroupID;
    uvec3 temp4 = gl_LocalInvocationID;
    uvec3 temp5 = gl_GlobalInvocationID;
    uint  temp6 = gl_LocalInvocationIndex;
    imageStore(imageOut, ivec2(gl_LocalInvocationIndex, 0), uvec4(temp1 + temp2 + temp3 + temp4 + temp5, temp6));
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
}

// Access part compute shader special variables.
TEST_P(ComputeShaderTest, AccessPartSpecialVariables)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=4, local_size_y=3, local_size_z=2) in;
layout(rgba32ui) uniform highp writeonly uimage2D imageOut;
void main()
{
    uvec3 temp1 = gl_WorkGroupSize;
    uvec3 temp2 = gl_WorkGroupID;
    uint  temp3 = gl_LocalInvocationIndex;
    imageStore(imageOut, ivec2(gl_LocalInvocationIndex, 0), uvec4(temp1 + temp2, temp3));
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
}

// Use glDispatchCompute to define work group count.
TEST_P(ComputeShaderTest, DispatchCompute)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=4, local_size_y=3, local_size_z=2) in;
layout(rgba32ui) uniform highp writeonly uimage2D imageOut;
void main()
{
    uvec3 temp = gl_NumWorkGroups;
    imageStore(imageOut, ivec2(gl_GlobalInvocationID.xy), uvec4(temp, 0u));
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);

    GLTexture texture;
    createMockOutputImage(texture, GL_RGBA32UI, 4, 3);

    glUseProgram(program);
    glDispatchCompute(8, 4, 2);
    EXPECT_GL_NO_ERROR();
}

// Binds a storage buffer to slot 0, then binds a storage image to slot 0, then buffer again.
TEST_P(ComputeShaderTest, BufferImageBuffer)
{
    constexpr char kCS0[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(binding = 0, offset = 4) uniform atomic_uint ac[2];
void main()
{
    atomicCounterIncrement(ac[0]);
    atomicCounterDecrement(ac[1]);
})";

    ANGLE_GL_COMPUTE_PROGRAM(program0, kCS0);
    glUseProgram(program0);

    unsigned int bufferData[3] = {11u, 4u, 4u};
    GLBuffer atomicCounterBuffer;
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, atomicCounterBuffer);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(bufferData), bufferData, GL_STATIC_DRAW);

    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, atomicCounterBuffer);

    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    auto mappedBuffer = MapBufferRange<const GLuint>(GL_ATOMIC_COUNTER_BUFFER, 0,
                                                     sizeof(bufferData), GL_MAP_READ_BIT);
    ASSERT_FALSE(mappedBuffer.empty());
    Span(bufferData).copy_from(mappedBuffer);
    glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);

    EXPECT_EQ(11u, bufferData[0]);
    EXPECT_EQ(5u, bufferData[1]);
    EXPECT_EQ(3u, bufferData[2]);

    constexpr char kCS1[] = R"(#version 310 es
layout(local_size_x=4, local_size_y=3, local_size_z=2) in;
layout(rgba32ui) uniform highp writeonly uimage2D imageOut;
void main()
{
    uvec3 temp = gl_NumWorkGroups;
    imageStore(imageOut, ivec2(gl_GlobalInvocationID.xy), uvec4(temp, 0u));
})";

    ANGLE_GL_COMPUTE_PROGRAM(program1, kCS1);

    GLTexture texture;
    createMockOutputImage(texture, GL_RGBA32UI, 4, 3);

    glUseProgram(program1);
    glDispatchCompute(8, 4, 2);

    glMemoryBarrier(GL_ATOMIC_COUNTER_BARRIER_BIT);
    glUseProgram(program0);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    mappedBuffer = MapBufferRange<const GLuint>(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(bufferData),
                                                GL_MAP_READ_BIT);
    ASSERT_FALSE(mappedBuffer.empty());
    Span(bufferData).copy_from(mappedBuffer);
    glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);

    EXPECT_EQ(11u, bufferData[0]);
    EXPECT_EQ(6u, bufferData[1]);
    EXPECT_EQ(2u, bufferData[2]);

    EXPECT_GL_NO_ERROR();
}

// Test that the buffer written to by imageStore() in the CS does not race with writing to the
// buffer when it's mapped.
TEST_P(ComputeShaderTest, BufferImageBufferMapWrite)
{
    constexpr char kCS0[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(binding = 0, offset = 4) uniform atomic_uint ac[2];
void main()
{
    atomicCounterIncrement(ac[0]);
    atomicCounterDecrement(ac[1]);
})";

    ANGLE_GL_COMPUTE_PROGRAM(program0, kCS0);
    glUseProgram(program0);

    std::array expectedBufferData = {11u, 4u, 4u};
    std::array bufferData         = expectedBufferData;

    GLBuffer atomicCounterBuffer;
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, atomicCounterBuffer);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(bufferData), bufferData.data(), GL_STATIC_DRAW);

    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, atomicCounterBuffer);

    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    auto mappedBuffer = MapBufferRange<const GLuint>(GL_ATOMIC_COUNTER_BUFFER, 0,
                                                     sizeof(bufferData), GL_MAP_READ_BIT);
    ASSERT_FALSE(mappedBuffer.empty());
    Span(bufferData).copy_from(mappedBuffer);
    glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);

    EXPECT_EQ(11u, bufferData[0]);
    EXPECT_EQ(5u, bufferData[1]);
    EXPECT_EQ(3u, bufferData[2]);

    constexpr char kCS1[] = R"(#version 310 es
layout(local_size_x=4, local_size_y=3, local_size_z=2) in;
layout(rgba32ui) uniform highp writeonly uimage2D imageOut;
void main()
{
    uvec3 temp = gl_NumWorkGroups;
    imageStore(imageOut, ivec2(gl_GlobalInvocationID.xy), uvec4(temp, 0u));
})";

    ANGLE_GL_COMPUTE_PROGRAM(program1, kCS1);

    GLTexture texture;
    createMockOutputImage(texture, GL_RGBA32UI, 4, 3);

    glUseProgram(program1);
    glDispatchCompute(8, 4, 2);

    glMemoryBarrier(GL_ATOMIC_COUNTER_BARRIER_BIT);
    glUseProgram(program0);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    auto writeBuffer = MapBufferRange<GLuint>(GL_ATOMIC_COUNTER_BUFFER, 0,
                                              sizeof(expectedBufferData), GL_MAP_WRITE_BIT);
    ASSERT_FALSE(writeBuffer.empty());
    writeBuffer.copy_from(Span(expectedBufferData));
    glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);

    // Force the CS imageStore() writes to the buffer to complete.
    glFinish();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    mappedBuffer = MapBufferRange<const GLuint>(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(bufferData),
                                                GL_MAP_READ_BIT);
    ASSERT_FALSE(mappedBuffer.empty());
    Span(bufferData).copy_from(mappedBuffer);
    glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);

    EXPECT_EQ(expectedBufferData[0], bufferData[0]);
    EXPECT_EQ(expectedBufferData[1], bufferData[1]);
    EXPECT_EQ(expectedBufferData[2], bufferData[2]);

    EXPECT_GL_NO_ERROR();
}

// Test that binds UAV with type image to slot 0, then binds UAV with type buffer to slot 0.
TEST_P(ComputeShaderTest, ImageAtomicCounterBuffer)
{
    // Flaky hang. http://anglebug.com/40644695
    ANGLE_SKIP_TEST_IF(IsWindows() && IsNVIDIA() && IsDesktopOpenGL());

    constexpr char kCS0[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(r32ui, binding = 0) writeonly uniform highp uimage2D uImage[2];
void main()
{
    imageStore(uImage[0], ivec2(gl_LocalInvocationIndex, gl_WorkGroupID.x), uvec4(100, 0,
0, 0));
    imageStore(uImage[1], ivec2(gl_LocalInvocationIndex, gl_WorkGroupID.x), uvec4(100, 0,
0, 0));
})";

    ANGLE_GL_COMPUTE_PROGRAM(program0, kCS0);
    glUseProgram(program0);
    int width = 1, height = 1;
    GLuint inputValues[] = {200};
    GLTexture mTexture[2];
    glBindTexture(GL_TEXTURE_2D, mTexture[0]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, width, height);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    inputValues);

    glBindTexture(GL_TEXTURE_2D, mTexture[1]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, width, height);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    inputValues);

    glBindImageTexture(0, mTexture[0], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
    glBindImageTexture(1, mTexture[1], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);

    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    constexpr char kCS1[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(binding = 0, offset = 4) uniform atomic_uint ac[2];
void main()
{
    atomicCounterIncrement(ac[0]);
    atomicCounterDecrement(ac[1]);
})";

    ANGLE_GL_COMPUTE_PROGRAM(program1, kCS1);

    unsigned int bufferData[3] = {11u, 4u, 4u};
    GLBuffer atomicCounterBuffer;
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, atomicCounterBuffer);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(bufferData), bufferData, GL_STATIC_DRAW);

    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, atomicCounterBuffer);

    glUseProgram(program1);
    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    auto mappedBuffer = MapBufferRange<const GLuint>(GL_ATOMIC_COUNTER_BUFFER, 0,
                                                     sizeof(bufferData), GL_MAP_READ_BIT);
    ASSERT_FALSE(mappedBuffer.empty());
    Span(bufferData).copy_from(mappedBuffer);
    glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);

    EXPECT_EQ(11u, bufferData[0]);
    EXPECT_EQ(5u, bufferData[1]);
    EXPECT_EQ(3u, bufferData[2]);

    EXPECT_GL_NO_ERROR();
}

// Test that binds UAV with type image to slot 0, then binds UAV with type buffer to slot 0.
TEST_P(ComputeShaderTest, ImageShaderStorageBuffer)
{
    constexpr char kCS0[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(r32ui, binding = 0) writeonly uniform highp uimage2D uImage[2];
void main()
{
    imageStore(uImage[0], ivec2(gl_LocalInvocationIndex, gl_WorkGroupID.x), uvec4(100, 0,
0, 0));
    imageStore(uImage[1], ivec2(gl_LocalInvocationIndex, gl_WorkGroupID.x), uvec4(100, 0,
0, 0));
})";

    ANGLE_GL_COMPUTE_PROGRAM(program0, kCS0);
    glUseProgram(program0);
    int width = 1, height = 1;
    GLuint inputValues[] = {200};
    GLTexture mTexture[2];
    glBindTexture(GL_TEXTURE_2D, mTexture[0]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, width, height);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    inputValues);

    glBindTexture(GL_TEXTURE_2D, mTexture[1]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, width, height);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    inputValues);

    glBindImageTexture(0, mTexture[0], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
    glBindImageTexture(1, mTexture[1], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);

    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    constexpr char kCS1[] =
        R"(#version 310 es
 layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
 layout(std140, binding = 0) buffer blockOut {
     uvec2 data;
 } instanceOut;
 layout(std140, binding = 1) buffer blockIn {
     uvec2 data;
 } instanceIn;
 void main()
 {
     instanceOut.data = instanceIn.data;
 }
 )";

    ANGLE_GL_COMPUTE_PROGRAM(program1, kCS1);

    constexpr unsigned int kBufferSize              = 2;
    constexpr std::array<unsigned int, kBufferSize> kBufferData = {10, 20};

    GLBuffer blockIn;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, blockIn);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(kBufferData), kBufferData.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, blockIn);

    GLBuffer blockOut;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, blockOut);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(kBufferData), nullptr, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, blockOut);

    glUseProgram(program1);
    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, blockOut);
    auto mapped = MapBufferRange<const unsigned int>(GL_SHADER_STORAGE_BUFFER, 0,
                                                     sizeof(kBufferData), GL_MAP_READ_BIT);
    EXPECT_EQ(Span(kBufferData), mapped);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
}

// Basic test for DispatchComputeIndirect.
TEST_P(ComputeShaderTest, DispatchComputeIndirect)
{
    // Flaky crash on teardown, see http://anglebug.com/40096579
    ANGLE_SKIP_TEST_IF(IsD3D11() && IsIntel() && IsWindows());

    const char kCSSource[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(r32ui, binding = 0) uniform highp uimage2D uImage;
void main()
{
    imageStore(uImage, ivec2(gl_WorkGroupID.x, gl_WorkGroupID.y), uvec4(100, 0, 0, 0));
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCSSource);
    glUseProgram(program);
    const int kWidth = 4, kHeight = 6;
    GLuint inputValues[kWidth][kHeight] = {};

    GLBuffer buffer;
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, buffer);
    GLuint params[] = {kWidth, kHeight, 1};
    glBufferData(GL_DISPATCH_INDIRECT_BUFFER, sizeof(params), params, GL_STATIC_DRAW);

    GLTexture texture;
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, kWidth, kHeight);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    inputValues);
    EXPECT_GL_NO_ERROR();

    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();

    glDispatchComputeIndirect(0);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);

    std::array<std::array<GLuint, kHeight>, kWidth> outputValues;

    GLFramebuffer framebuffer;
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
    EXPECT_GL_FRAMEBUFFER_COMPLETE(GL_READ_FRAMEBUFFER);
    EXPECT_GL_NO_ERROR();

    glReadPixels(0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT, outputValues.data());
    EXPECT_GL_NO_ERROR();

    constexpr GLuint kExpectedValue = 100u;
    for (int x = 0; x < kWidth; x++)
    {
        for (int y = 0; y < kHeight; y++)
        {
            EXPECT_EQ(kExpectedValue, outputValues[x][y]);
        }
    }
}

// Test that uploading data to buffer that's in use then using it as indirect buffer works.
TEST_P(ComputeShaderTest, UseAsUBOThenUpdateThenDispatchComputeIndirect)
{
    // Flaky crash on teardown, see http://anglebug.com/40096579
    ANGLE_SKIP_TEST_IF(IsD3D11() && IsIntel() && IsWindows());

    constexpr GLsizei kWidth = 4, kHeight = 6;

    const std::array<uint32_t, 4> kInitialData = {1, 2, 3, 4};
    const std::array<uint32_t, 4> kUpdateData  = {kWidth, kHeight, 1, 0};

    GLBuffer buffer;
    glBindBuffer(GL_UNIFORM_BUFFER, buffer);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(kInitialData), kInitialData.data(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, buffer);
    EXPECT_GL_NO_ERROR();

    constexpr char kVerifyUBO[] = R"(#version 310 es
precision mediump float;
layout(binding = 0) uniform block {
    uvec4 data;
} ubo;
out vec4 colorOut;
void main()
{
    if (all(equal(ubo.data, uvec4(1, 2, 3, 4))))
        colorOut = vec4(0, 1.0, 0, 1.0);
    else
        colorOut = vec4(1.0, 0, 0, 1.0);
})";

    ANGLE_GL_PROGRAM(verifyUbo, essl31_shaders::vs::Simple(), kVerifyUBO);
    drawQuad(verifyUbo, essl31_shaders::PositionAttrib(), 0.5);
    EXPECT_GL_NO_ERROR();

    // Update buffer data
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(kInitialData), kUpdateData.data());
    EXPECT_GL_NO_ERROR();

    const char kCS[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(r32ui, binding = 0) uniform highp uimage2D uImage;
void main()
{
    imageStore(uImage, ivec2(gl_WorkGroupID.x, gl_WorkGroupID.y), uvec4(100, 0, 0, 0));
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    glUseProgram(program);

    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, buffer);

    const std::vector<GLuint> inputValues(kWidth * kHeight, 0);

    GLTexture texture;
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, kWidth, kHeight);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    inputValues.data());
    EXPECT_GL_NO_ERROR();

    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();

    glDispatchComputeIndirect(0);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);

    EXPECT_PIXEL_COLOR_EQ(0, 0, GLColor::green);

    std::array<std::array<GLuint, kHeight>, kWidth> outputValues;

    GLFramebuffer framebuffer;
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
    EXPECT_GL_FRAMEBUFFER_COMPLETE(GL_READ_FRAMEBUFFER);
    EXPECT_GL_NO_ERROR();

    glReadPixels(0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT, outputValues.data());
    EXPECT_GL_NO_ERROR();

    constexpr GLuint kExpectedValue = 100u;
    for (int x = 0; x < kWidth; x++)
    {
        for (int y = 0; y < kHeight; y++)
        {
            EXPECT_EQ(kExpectedValue, outputValues[x][y]);
        }
    }
}

// Use image uniform to write texture in compute shader, and verify the content is expected.
TEST_P(ComputeShaderTest, BindImageTexture)
{
    GLTexture mTexture[2];
    GLFramebuffer mFramebuffer;
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(r32ui, binding = 0) writeonly uniform highp uimage2D uImage[2];
void main()
{
    imageStore(uImage[0], ivec2(gl_LocalInvocationIndex, gl_WorkGroupID.x), uvec4(100, 0,
0, 0));
    imageStore(uImage[1], ivec2(gl_LocalInvocationIndex, gl_WorkGroupID.x), uvec4(100, 0,
0, 0));
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    glUseProgram(program);
    int width = 1, height = 1;
    GLuint inputValues[] = {200};

    glBindTexture(GL_TEXTURE_2D, mTexture[0]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, width, height);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    inputValues);
    EXPECT_GL_NO_ERROR();

    glBindImageTexture(0, mTexture[0], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();

    glBindTexture(GL_TEXTURE_2D, mTexture[1]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, width, height);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    inputValues);
    EXPECT_GL_NO_ERROR();

    glBindImageTexture(1, mTexture[1], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();

    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
    glUseProgram(0);
    std::array<std::array<GLuint, 1>, 2> outputValues;
    GLuint expectedValue = 100;
    glBindFramebuffer(GL_READ_FRAMEBUFFER, mFramebuffer);

    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mTexture[0],
                           0);
    EXPECT_GL_NO_ERROR();
    glReadPixels(0, 0, width, height, GL_RED_INTEGER, GL_UNSIGNED_INT, outputValues[0].data());
    EXPECT_GL_NO_ERROR();

    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mTexture[1],
                           0);
    EXPECT_GL_NO_ERROR();
    glReadPixels(0, 0, width, height, GL_RED_INTEGER, GL_UNSIGNED_INT, outputValues[1].data());
    EXPECT_GL_NO_ERROR();

    for (int i = 0; i < width * height; i++)
    {
        EXPECT_EQ(expectedValue, outputValues[0][i]);
        EXPECT_EQ(expectedValue, outputValues[1][i]);
    }
}

// Test that binding a 2D slice of a 3D texture works with compute shader
TEST_P(ComputeShaderTest, BindImageTexture3D)
{
    GLTexture mTexture[2];
    GLFramebuffer mFramebuffer;
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(r32ui, binding = 0) writeonly uniform highp uimage2D uImage[2];
void main()
{
    imageStore(uImage[0], ivec2(gl_LocalInvocationIndex, gl_WorkGroupID.x), uvec4(100, 0,
0, 0));
    imageStore(uImage[1], ivec2(gl_LocalInvocationIndex, gl_WorkGroupID.x), uvec4(100, 0,
0, 0));
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    glUseProgram(program);
    int width = 1, height = 1, depth = 1;
    GLuint inputValues[] = {200};

    glBindTexture(GL_TEXTURE_3D, mTexture[0]);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_R32UI, width, height, depth);
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, width, height, depth, GL_RED_INTEGER,
                    GL_UNSIGNED_INT, inputValues);
    EXPECT_GL_NO_ERROR();

    glBindImageTexture(0, mTexture[0], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();

    glBindTexture(GL_TEXTURE_3D, mTexture[1]);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_R32UI, width, height, depth);
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, width, height, depth, GL_RED_INTEGER,
                    GL_UNSIGNED_INT, inputValues);
    EXPECT_GL_NO_ERROR();

    glBindImageTexture(1, mTexture[1], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();

    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
    glUseProgram(0);
    std::array<std::array<GLuint, 1>, 2> outputValues;
    GLuint expectedValue = 100;
    glBindFramebuffer(GL_READ_FRAMEBUFFER, mFramebuffer);

    glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, mTexture[0], 0, 0);
    EXPECT_GL_NO_ERROR();
    glReadPixels(0, 0, width, height, GL_RED_INTEGER, GL_UNSIGNED_INT, outputValues[0].data());
    EXPECT_GL_NO_ERROR();

    glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, mTexture[1], 0, 0);
    EXPECT_GL_NO_ERROR();
    glReadPixels(0, 0, width, height, GL_RED_INTEGER, GL_UNSIGNED_INT, outputValues[1].data());
    EXPECT_GL_NO_ERROR();

    for (int i = 0; i < width * height; i++)
    {
        EXPECT_EQ(expectedValue, outputValues[0][i]);
        EXPECT_EQ(expectedValue, outputValues[1][i]);
    }
}

// When declare a image array without a binding qualifier, all elements are bound to unit zero.
TEST_P(ComputeShaderTest, ImageArrayWithoutBindingQualifier)
{
    ANGLE_SKIP_TEST_IF(IsD3D11());

    // TODO(xinghua.cao@intel.com): On AMD desktop OpenGL, bind two image variables to unit 0,
    // only one variable is valid.
    ANGLE_SKIP_TEST_IF(IsAMD() && IsDesktopOpenGL());

    GLTexture mTexture;
    GLFramebuffer mFramebuffer;
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(r32ui) writeonly uniform highp uimage2D uImage[2];
void main()
{
    imageStore(uImage[0], ivec2(gl_LocalInvocationIndex, 0), uvec4(100, 0, 0, 0));
    imageStore(uImage[1], ivec2(gl_LocalInvocationIndex, 1), uvec4(100, 0, 0, 0));
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    glUseProgram(program);
    constexpr int kTextureWidth = 1, kTextureHeight = 2;
    GLuint inputValues[] = {200, 200};

    glBindTexture(GL_TEXTURE_2D, mTexture);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, kTextureWidth, kTextureHeight);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kTextureWidth, kTextureHeight, GL_RED_INTEGER,
                    GL_UNSIGNED_INT, inputValues);
    EXPECT_GL_NO_ERROR();

    glBindImageTexture(0, mTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
    glUseProgram(0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, mFramebuffer);

    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mTexture, 0);
    std::array<GLuint, kTextureWidth * kTextureHeight> outputValues;
    glReadPixels(0, 0, kTextureWidth, kTextureHeight, GL_RED_INTEGER, GL_UNSIGNED_INT,
                 outputValues.data());
    EXPECT_GL_NO_ERROR();

    GLuint expectedValue = 100;
    for (int i = 0; i < kTextureWidth * kTextureHeight; i++)
    {
        EXPECT_EQ(expectedValue, outputValues[i]);
    }
}

// When an image array is declared without a binding qualifier, all elements are bound to unit zero.
// Check that the unused uniform image array element does not cause any corruption. Checks for a bug
// where unused element could make the whole array seem as unused.
TEST_P(ComputeShaderTest, ImageArrayUnusedElement)
{
    ANGLE_SKIP_TEST_IF(IsD3D11());

    // TODO(xinghua.cao@intel.com): On AMD desktop OpenGL, bind two image variables to unit 0,
    // only one variable is valid.
    ANGLE_SKIP_TEST_IF(IsAMD() && IsDesktopOpenGL());

    // Vulkan is currently unable to handle unbound image units in compute shaders.
    // http://anglebug.com/42263596
    ANGLE_SKIP_TEST_IF(IsVulkan());

    GLFramebuffer framebuffer;
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(r32ui, binding=0) writeonly uniform highp uimage2D uOut;
layout(r32ui, binding=1) readonly uniform highp uimage2D uIn[2];

void main()
{
    uvec4 inValue = imageLoad(uIn[0], ivec2(gl_LocalInvocationID.xy));
    imageStore(uOut, ivec2(gl_LocalInvocationIndex, 0), inValue);
    imageStore(uOut, ivec2(gl_LocalInvocationIndex, 1), inValue);
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    glUseProgram(program);
    constexpr int kTextureWidth = 1, kTextureHeight = 2;
    GLuint inputValues[] = {100, 100};
    GLTexture in;
    glBindTexture(GL_TEXTURE_2D, in);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, kTextureWidth, kTextureHeight);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kTextureWidth, kTextureHeight, GL_RED_INTEGER,
                    GL_UNSIGNED_INT, inputValues);
    EXPECT_GL_NO_ERROR();
    glBindImageTexture(1, in, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI);

    GLuint initValues[] = {111, 111};
    GLTexture out;
    glBindTexture(GL_TEXTURE_2D, out);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, kTextureWidth, kTextureHeight);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kTextureWidth, kTextureHeight, GL_RED_INTEGER,
                    GL_UNSIGNED_INT, initValues);
    EXPECT_GL_NO_ERROR();

    glBindImageTexture(0, out, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
    glUseProgram(0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);

    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, out, 0);
    std::array<GLuint, kTextureWidth * kTextureHeight> outputValues;
    glReadPixels(0, 0, kTextureWidth, kTextureHeight, GL_RED_INTEGER, GL_UNSIGNED_INT,
                 outputValues.data());
    EXPECT_GL_NO_ERROR();

    GLuint expectedValue = 100;
    for (int i = 0; i < kTextureWidth * kTextureHeight; i++)
    {
        EXPECT_EQ(expectedValue, outputValues[i]);
    }
}
// imageLoad functions
TEST_P(ComputeShaderTest, ImageLoad)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=8) in;
layout(rgba8) uniform highp readonly image2D mImage2DInput;
layout(rgba16i) uniform highp readonly iimageCube mImageCubeInput;
layout(rgba32ui) uniform highp readonly uimage3D mImage3DInput;
layout(r32i) uniform highp writeonly iimage2D imageOut;
void main()
{
    vec4 result2d = imageLoad(mImage2DInput, ivec2(gl_LocalInvocationID.xy));
    ivec4 resultCube = imageLoad(mImageCubeInput, ivec3(gl_LocalInvocationID.xyz));
    uvec4 result3d = imageLoad(mImage3DInput, ivec3(gl_LocalInvocationID.xyz));
    imageStore(imageOut, ivec2(gl_LocalInvocationIndex, 0), ivec4(result2d) + resultCube + ivec4(result3d));
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    EXPECT_GL_NO_ERROR();
}

// imageStore functions
TEST_P(ComputeShaderTest, ImageStore)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=8) in;
layout(rgba16f) uniform highp writeonly imageCube mImageCubeOutput;
layout(r32f) uniform highp writeonly image3D mImage3DOutput;
layout(rgba8ui) uniform highp writeonly uimage2DArray mImage2DArrayOutput;
void main()
{
    imageStore(mImageCubeOutput, ivec3(gl_LocalInvocationID.xyz), vec4(0.0));
    imageStore(mImage3DOutput, ivec3(gl_LocalInvocationID.xyz), vec4(0.0));
    imageStore(mImage2DArrayOutput, ivec3(gl_LocalInvocationID.xyz), uvec4(0));
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    EXPECT_GL_NO_ERROR();
}

// imageSize functions
TEST_P(ComputeShaderTest, ImageSize)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=8) in;
layout(rgba8) uniform highp readonly imageCube mImageCubeInput;
layout(r32i) uniform highp readonly iimage2D mImage2DInput;
layout(rgba16ui) uniform highp readonly uimage2DArray mImage2DArrayInput;
layout(r32i) uniform highp writeonly iimage2D imageOut;
void main()
{
    ivec2 sizeCube = imageSize(mImageCubeInput);
    ivec2 size2D = imageSize(mImage2DInput);
    ivec3 size2DArray = imageSize(mImage2DArrayInput);
    imageStore(imageOut, ivec2(gl_LocalInvocationIndex, 0), ivec4(sizeCube, size2D.x, size2DArray.x));
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    EXPECT_GL_NO_ERROR();
}

// Test that texelFetch works well in compute shader.
TEST_P(ComputeShaderTest, TexelFetchFunction)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=16, local_size_y=12) in;
precision highp usampler2D;
uniform usampler2D tex;
layout(std140, binding = 0) buffer buf {
    uint outData[12][16];
};

void main()
{
    uint x = gl_LocalInvocationID.x;
    uint y = gl_LocalInvocationID.y;
    outData[y][x] = texelFetch(tex, ivec2(x, y), 0).x;
})";

    constexpr unsigned int kWidth  = 16;
    constexpr unsigned int kHeight = 12;
    GLTexture tex;
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, kWidth, kHeight);
    std::array<std::array<GLuint, kWidth>, kHeight> texels = {};
    for (unsigned int y = 0; y < kHeight; ++y)
    {
        for (unsigned int x = 0; x < kWidth; ++x)
        {
            texels[y][x] = x + y * kWidth;
        }
    }
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    texels.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    // The array stride are rounded up to the base alignment of a vec4 for std140 layout.
    constexpr unsigned int kArrayStride = 16;
    GLBuffer ssbo;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, kWidth * kHeight * kArrayStride, nullptr,
                 GL_STREAM_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    EXPECT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    glUseProgram(program);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glUniform1i(glGetUniformLocation(program, "tex"), 0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);

    glDispatchCompute(1, 1, 1);

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    auto mapped = MapBufferRange<const GLuint>(GL_SHADER_STORAGE_BUFFER, 0,
                                               kWidth * kHeight * kArrayStride, GL_MAP_READ_BIT);
    EXPECT_GL_NO_ERROR();
    ASSERT_FALSE(mapped.empty());
    for (unsigned int idx = 0; idx < kWidth * kHeight; idx++)
    {
        EXPECT_EQ(idx, mapped[idx * kArrayStride / sizeof(GLuint)]);
    }
}

// Test that texture function works well in compute shader.
TEST_P(ComputeShaderTest, TextureFunction)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=16, local_size_y=16) in;
precision highp usampler2D;
uniform usampler2D tex;
layout(std140, binding = 0) buffer buf {
    uint outData[16][16];
};

void main()
{
    uint x = gl_LocalInvocationID.x;
    uint y = gl_LocalInvocationID.y;
    float xCoord = float(x) / float(16);
    float yCoord = float(y) / float(16);
    outData[y][x] = texture(tex, vec2(xCoord, yCoord)).x;
})";

    constexpr unsigned int kWidth  = 16;
    constexpr unsigned int kHeight = 16;
    GLTexture tex;
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, kWidth, kHeight);
    std::array<std::array<GLuint, kWidth>, kHeight> texels = {};
    for (unsigned int y = 0; y < kHeight; ++y)
    {
        for (unsigned int x = 0; x < kWidth; ++x)
        {
            texels[y][x] = x + y * kWidth;
        }
    }
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    texels.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    // The array stride are rounded up to the base alignment of a vec4 for std140 layout.
    constexpr unsigned int kArrayStride = 16;
    GLBuffer ssbo;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, kWidth * kHeight * kArrayStride, nullptr,
                 GL_STREAM_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    EXPECT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    glUseProgram(program);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glUniform1i(glGetUniformLocation(program, "tex"), 0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);

    glDispatchCompute(1, 1, 1);

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    auto mapped = MapBufferRange<const GLuint>(GL_SHADER_STORAGE_BUFFER, 0,
                                               kWidth * kHeight * kArrayStride, GL_MAP_READ_BIT);
    EXPECT_GL_NO_ERROR();
    ASSERT_FALSE(mapped.empty());
    for (unsigned int idx = 0; idx < kWidth * kHeight; idx++)
    {
        EXPECT_EQ(idx, mapped[idx * kArrayStride / sizeof(GLuint)]);
    }
}

// Test mixed use of sampler and image.
TEST_P(ComputeShaderTest, SamplingAndImageReadWrite)
{
    GLTexture texture[3];
    GLFramebuffer framebuffer;
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(r32ui, binding = 0) readonly uniform highp uimage2D uImage_1;
layout(r32ui, binding = 1) writeonly uniform highp uimage2D uImage_2;
precision highp usampler2D;
uniform usampler2D tex;
void main()
{
    uvec4 value_1 = texelFetch(tex, ivec2(gl_LocalInvocationID.xy), 0);
    uvec4 value_2 = imageLoad(uImage_1, ivec2(gl_LocalInvocationID.xy));
    imageStore(uImage_2, ivec2(gl_LocalInvocationID.xy), value_1 + value_2);
})";

    constexpr int kWidth = 1, kHeight = 1;
    constexpr GLuint kInputValues[3][1] = {{50}, {100}, {20}};

    glBindTexture(GL_TEXTURE_2D, texture[0]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, kWidth, kHeight);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    kInputValues[0]);
    glBindTexture(GL_TEXTURE_2D, texture[2]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, kWidth, kHeight);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    kInputValues[2]);
    EXPECT_GL_NO_ERROR();
    glBindTexture(GL_TEXTURE_2D, 0);

    glBindTexture(GL_TEXTURE_2D, texture[1]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, kWidth, kHeight);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    kInputValues[1]);

    EXPECT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    glUseProgram(program);

    glBindImageTexture(0, texture[0], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI);
    glBindImageTexture(1, texture[2], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();

    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
    std::array<GLuint, kWidth * kHeight> outputValues;
    constexpr GLuint expectedValue = 150;
    glUseProgram(0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);

    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture[2], 0);
    EXPECT_GL_NO_ERROR();
    glReadPixels(0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT, outputValues.data());
    EXPECT_GL_NO_ERROR();

    for (int i = 0; i < kWidth * kHeight; i++)
    {
        EXPECT_EQ(expectedValue, outputValues[i]);
    }
}

// Use image uniform to read and write Texture2D in compute shader, and verify the contents.
TEST_P(ComputeShaderTest, BindImageTextureWithTexture2D)
{
    GLTexture texture[2];
    GLFramebuffer framebuffer;
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(r32ui, binding = 0) readonly uniform highp uimage2D uImage_1;
layout(r32ui, binding = 1) writeonly uniform highp uimage2D uImage_2;
void main()
{
    uvec4 value = imageLoad(uImage_1, ivec2(gl_LocalInvocationID.xy));
    imageStore(uImage_2, ivec2(gl_LocalInvocationID.xy), value);
})";

    constexpr int kWidth = 1, kHeight = 1;
    constexpr GLuint kInputValues[2][1] = {{200}, {100}};

    glBindTexture(GL_TEXTURE_2D, texture[0]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, kWidth, kHeight);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    kInputValues[0]);
    EXPECT_GL_NO_ERROR();

    glBindTexture(GL_TEXTURE_2D, texture[1]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, kWidth, kHeight);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    kInputValues[1]);
    EXPECT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    glUseProgram(program);

    glBindImageTexture(0, texture[0], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();

    glBindImageTexture(1, texture[1], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();

    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
    std::array<GLuint, kWidth * kHeight> outputValues;
    constexpr GLuint expectedValue = 200;
    glUseProgram(0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);

    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture[1], 0);
    EXPECT_GL_NO_ERROR();
    glReadPixels(0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT, outputValues.data());
    EXPECT_GL_NO_ERROR();

    for (int i = 0; i < kWidth * kHeight; i++)
    {
        EXPECT_EQ(expectedValue, outputValues[i]);
    }
}

// Use image uniform to read and write Texture2D with non-zero base in compute shader, and verify
// the contents.
TEST_P(ComputeShaderTest, BindImageTextureWithNonZeroBaseTexture2D)
{
    GLTexture texture[2];
    GLFramebuffer framebuffer;
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(r32ui, binding = 0) readonly uniform highp uimage2D uImage_1;
layout(r32ui, binding = 1) writeonly uniform highp uimage2D uImage_2;
void main()
{
    uvec4 value = imageLoad(uImage_1, ivec2(gl_LocalInvocationID.xy));
    imageStore(uImage_2, ivec2(gl_LocalInvocationID.xy), value);
})";

    constexpr int kWidth = 1, kHeight = 1;
    constexpr GLuint kInputValues[2][1] = {{200}, {100}};

    glBindTexture(GL_TEXTURE_2D, texture[0]);
    glTexStorage2D(GL_TEXTURE_2D, 2, GL_R32UI, kWidth * 2, kHeight * 2);
    glTexSubImage2D(GL_TEXTURE_2D, 1, 0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    kInputValues[0]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 1);
    EXPECT_GL_NO_ERROR();

    glBindTexture(GL_TEXTURE_2D, texture[1]);
    glTexStorage2D(GL_TEXTURE_2D, 2, GL_R32UI, kWidth * 2, kHeight * 2);
    glTexSubImage2D(GL_TEXTURE_2D, 1, 0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    kInputValues[1]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 1);
    EXPECT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    glUseProgram(program);

    glBindImageTexture(0, texture[0], 1, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();

    glBindImageTexture(1, texture[1], 1, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();

    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
    std::array<GLuint, kWidth * kHeight> outputValues;
    constexpr GLuint expectedValue = 200;
    glUseProgram(0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);

    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture[1], 1);
    EXPECT_GL_NO_ERROR();
    glReadPixels(0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT, outputValues.data());
    EXPECT_GL_NO_ERROR();

    for (int i = 0; i < kWidth * kHeight; i++)
    {
        EXPECT_EQ(expectedValue, outputValues[i]) << "  at index: " << i;
    }
}

// Use image uniform to read and write Texture2DArray in compute shader, and verify the contents.
TEST_P(ComputeShaderTest, BindImageTextureWithTexture2DArray)
{
    GLTexture texture[2];
    GLFramebuffer framebuffer;
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=2, local_size_y=2, local_size_z=2) in;
layout(r32ui, binding = 0) readonly uniform highp uimage2DArray uImage_1;
layout(r32ui, binding = 1) writeonly uniform highp uimage2DArray uImage_2;
void main()
{
    uvec4 value = imageLoad(uImage_1, ivec3(gl_LocalInvocationID.xyz));
    imageStore(uImage_2, ivec3(gl_LocalInvocationID.xyz), value);
})";

    constexpr int kWidth = 1, kHeight = 1, kDepth = 2;
    constexpr GLuint kInputValues[2][2] = {{200, 200}, {100, 100}};

    glBindTexture(GL_TEXTURE_2D_ARRAY, texture[0]);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_R32UI, kWidth, kHeight, kDepth);
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, kWidth, kHeight, kDepth, GL_RED_INTEGER,
                    GL_UNSIGNED_INT, kInputValues[0]);
    EXPECT_GL_NO_ERROR();

    glBindTexture(GL_TEXTURE_2D_ARRAY, texture[1]);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_R32UI, kWidth, kHeight, kDepth);
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, kWidth, kHeight, kDepth, GL_RED_INTEGER,
                    GL_UNSIGNED_INT, kInputValues[1]);
    EXPECT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    glUseProgram(program);

    glBindImageTexture(0, texture[0], 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();

    glBindImageTexture(1, texture[1], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();

    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
    std::array<GLuint, kWidth * kHeight> outputValues;
    constexpr GLuint expectedValue = 200;
    glUseProgram(0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);

    glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture[1], 0, 0);
    glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, texture[1], 0, 1);
    EXPECT_GL_NO_ERROR();
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT, outputValues.data());
    EXPECT_GL_NO_ERROR();
    for (int i = 0; i < kWidth * kHeight; i++)
    {
        EXPECT_EQ(expectedValue, outputValues[i]);
    }
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    glReadPixels(0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT, outputValues.data());
    EXPECT_GL_NO_ERROR();
    for (int i = 0; i < kWidth * kHeight; i++)
    {
        EXPECT_EQ(expectedValue, outputValues[i]);
    }
}

// Use image uniform to read and write Texture2DArray with non-zero base in compute shader, and
// verify the contents.
TEST_P(ComputeShaderTest, BindImageTextureWithNonZeroBaseTexture2DArray)
{
    GLTexture texture[2];
    GLFramebuffer framebuffer;
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=2, local_size_y=2, local_size_z=2) in;
layout(r32ui, binding = 0) readonly uniform highp uimage2DArray uImage_1;
layout(r32ui, binding = 1) writeonly uniform highp uimage2DArray uImage_2;
void main()
{
    uvec4 value = imageLoad(uImage_1, ivec3(gl_LocalInvocationID.xyz));
    imageStore(uImage_2, ivec3(gl_LocalInvocationID.xyz), value);
})";

    constexpr int kWidth = 1, kHeight = 1, kDepth = 2;
    constexpr GLuint kInputValues[2][2] = {{200, 200}, {100, 100}};

    glBindTexture(GL_TEXTURE_2D_ARRAY, texture[0]);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 2, GL_R32UI, kWidth * 2, kHeight * 2, kDepth);
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 1, 0, 0, 0, kWidth, kHeight, kDepth, GL_RED_INTEGER,
                    GL_UNSIGNED_INT, kInputValues[0]);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BASE_LEVEL, 1);
    EXPECT_GL_NO_ERROR();

    glBindTexture(GL_TEXTURE_2D_ARRAY, texture[1]);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 2, GL_R32UI, kWidth * 2, kHeight * 2, kDepth);
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 1, 0, 0, 0, kWidth, kHeight, kDepth, GL_RED_INTEGER,
                    GL_UNSIGNED_INT, kInputValues[1]);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BASE_LEVEL, 1);
    EXPECT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    glUseProgram(program);

    glBindImageTexture(0, texture[0], 1, GL_TRUE, 0, GL_READ_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();

    glBindImageTexture(1, texture[1], 1, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();

    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
    std::array<GLuint, kWidth * kHeight> outputValues;
    constexpr GLuint expectedValue = 200;
    glUseProgram(0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);

    glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture[1], 1, 0);
    glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, texture[1], 1, 1);
    EXPECT_GL_NO_ERROR();
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT, outputValues.data());
    EXPECT_GL_NO_ERROR();
    for (int i = 0; i < kWidth * kHeight; i++)
    {
        EXPECT_EQ(expectedValue, outputValues[i]);
    }
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    glReadPixels(0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT, outputValues.data());
    EXPECT_GL_NO_ERROR();
    for (int i = 0; i < kWidth * kHeight; i++)
    {
        EXPECT_EQ(expectedValue, outputValues[i]);
    }
}

// Use image uniform to read and write Texture3D in compute shader, and verify the contents.
TEST_P(ComputeShaderTest, BindImageTextureWithTexture3D)
{
    GLTexture texture[2];
    GLFramebuffer framebuffer;
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=2) in;
layout(r32ui, binding = 0) readonly uniform highp uimage3D uImage_1;
layout(r32ui, binding = 1) writeonly uniform highp uimage3D uImage_2;
void main()
{
    uvec4 value = imageLoad(uImage_1, ivec3(gl_LocalInvocationID.xyz));
    imageStore(uImage_2, ivec3(gl_LocalInvocationID.xyz), value);
})";

    constexpr int kWidth = 1, kHeight = 1, kDepth = 2;
    constexpr GLuint kInputValues[2][2] = {{200, 200}, {100, 100}};

    glBindTexture(GL_TEXTURE_3D, texture[0]);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_R32UI, kWidth, kHeight, kDepth);
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, kWidth, kHeight, kDepth, GL_RED_INTEGER,
                    GL_UNSIGNED_INT, kInputValues[0]);
    EXPECT_GL_NO_ERROR();

    glBindTexture(GL_TEXTURE_3D, texture[1]);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_R32UI, kWidth, kHeight, kDepth);
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, kWidth, kHeight, kDepth, GL_RED_INTEGER,
                    GL_UNSIGNED_INT, kInputValues[1]);
    EXPECT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    glUseProgram(program);

    glBindImageTexture(0, texture[0], 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();

    glBindImageTexture(1, texture[1], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();

    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
    std::array<GLuint, kWidth * kHeight> outputValues;
    constexpr GLuint expectedValue = 200;
    glUseProgram(0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);

    glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture[1], 0, 0);
    glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, texture[1], 0, 1);
    EXPECT_GL_NO_ERROR();
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT, outputValues.data());
    EXPECT_GL_NO_ERROR();
    for (int i = 0; i < kWidth * kHeight; i++)
    {
        EXPECT_EQ(expectedValue, outputValues[i]);
    }
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    glReadPixels(0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT, outputValues.data());
    EXPECT_GL_NO_ERROR();
    for (int i = 0; i < kWidth * kHeight; i++)
    {
        EXPECT_EQ(expectedValue, outputValues[i]);
    }
}

// Use image uniform to read and write TextureCube in compute shader, and verify the contents.
TEST_P(ComputeShaderTest, BindImageTextureWithTextureCube)
{
    GLTexture texture[2];
    GLFramebuffer framebuffer;
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(r32ui, binding = 0) readonly uniform highp uimageCube uImage_1;
layout(r32ui, binding = 1) writeonly uniform highp uimageCube uImage_2;
void main()
{
    for (int i = 0; i < 6; i++)
    {
        uvec4 value = imageLoad(uImage_1, ivec3(gl_LocalInvocationID.xy, i));
        imageStore(uImage_2, ivec3(gl_LocalInvocationID.xy, i), value);
    }
})";

    constexpr int kWidth = 1, kHeight = 1;
    constexpr GLuint kInputValues[2][1] = {{200}, {100}};

    glBindTexture(GL_TEXTURE_CUBE_MAP, texture[0]);
    glTexStorage2D(GL_TEXTURE_CUBE_MAP, 1, GL_R32UI, kWidth, kHeight);
    for (GLenum face = GL_TEXTURE_CUBE_MAP_POSITIVE_X; face <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z;
         face++)
    {
        glTexSubImage2D(face, 0, 0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT,
                        kInputValues[0]);
    }
    EXPECT_GL_NO_ERROR();

    glBindTexture(GL_TEXTURE_CUBE_MAP, texture[1]);
    glTexStorage2D(GL_TEXTURE_CUBE_MAP, 1, GL_R32UI, kWidth, kHeight);
    for (GLenum face = GL_TEXTURE_CUBE_MAP_POSITIVE_X; face <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z;
         face++)
    {
        glTexSubImage2D(face, 0, 0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT,
                        kInputValues[1]);
    }
    EXPECT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    glUseProgram(program);

    glBindImageTexture(0, texture[0], 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();

    glBindImageTexture(1, texture[1], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();

    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
    std::array<GLuint, kWidth * kHeight> outputValues;
    constexpr GLuint expectedValue = 200;
    glUseProgram(0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);

    for (GLenum face = 0; face < 6; face++)
    {
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, texture[1], 0);
        EXPECT_GL_NO_ERROR();
        glReadPixels(0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT, outputValues.data());
        EXPECT_GL_NO_ERROR();

        for (int i = 0; i < kWidth * kHeight; i++)
        {
            EXPECT_EQ(expectedValue, outputValues[i]);
        }
    }
}

// Use image uniform to read and write one layer of Texture2DArray in compute shader, and verify the
// contents.
TEST_P(ComputeShaderTest, BindImageTextureWithOneLayerTexture2DArray)
{
    GLTexture texture[2];
    GLFramebuffer framebuffer;
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(r32ui, binding = 0) readonly uniform highp uimage2D uImage_1;
layout(r32ui, binding = 1) writeonly uniform highp uimage2D uImage_2;
void main()
{
    uvec4 value = imageLoad(uImage_1, ivec2(gl_LocalInvocationID.xy));
    imageStore(uImage_2, ivec2(gl_LocalInvocationID.xy), value);
})";

    constexpr int kWidth = 1, kHeight = 1, kDepth = 2;
    constexpr int kResultSize           = kWidth * kHeight;
    constexpr GLuint kInputValues[2][2] = {{200, 150}, {100, 50}};
    constexpr GLuint expectedValue_1    = 200;
    constexpr GLuint expectedValue_2    = 100;
    std::array<GLuint, kResultSize> outputValues;

    glBindTexture(GL_TEXTURE_2D_ARRAY, texture[0]);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_R32UI, kWidth, kHeight, kDepth);
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, kWidth, kHeight, kDepth, GL_RED_INTEGER,
                    GL_UNSIGNED_INT, kInputValues[0]);
    EXPECT_GL_NO_ERROR();

    glBindTexture(GL_TEXTURE_2D_ARRAY, texture[1]);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_R32UI, kWidth, kHeight, kDepth);
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, kWidth, kHeight, kDepth, GL_RED_INTEGER,
                    GL_UNSIGNED_INT, kInputValues[1]);
    EXPECT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    glUseProgram(program);

    glBindImageTexture(0, texture[0], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();
    glBindImageTexture(1, texture[1], 0, GL_FALSE, 1, GL_WRITE_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();
    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
    glUseProgram(0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
    glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture[1], 0, 0);
    glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, texture[1], 0, 1);
    EXPECT_GL_NO_ERROR();
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT, outputValues.data());
    EXPECT_GL_NO_ERROR();
    for (int i = 0; i < kResultSize; i++)
    {
        EXPECT_EQ(expectedValue_2, outputValues[i]);
    }
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    glReadPixels(0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT, outputValues.data());
    EXPECT_GL_NO_ERROR();
    for (int i = 0; i < kResultSize; i++)
    {
        EXPECT_EQ(expectedValue_1, outputValues[i]);
    }
}

// Use image uniform to read and write one layer of Texture3D in compute shader, and verify the
// contents.
TEST_P(ComputeShaderTest, BindImageTextureWithOneLayerTexture3D)
{
    // Vulkan validation error creating a 2D image view of a 3D image layer.
    // http://anglebug.com/42262531
    ANGLE_SKIP_TEST_IF(IsVulkan());

    GLTexture texture[2];
    GLFramebuffer framebuffer;
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(r32ui, binding = 0) readonly uniform highp uimage2D uImage_1;
layout(r32ui, binding = 1) writeonly uniform highp uimage2D uImage_2;
void main()
{
    uvec4 value = imageLoad(uImage_1, ivec2(gl_LocalInvocationID.xy));
    imageStore(uImage_2, ivec2(gl_LocalInvocationID.xy), value);
})";

    constexpr int kWidth = 1, kHeight = 1, kDepth = 2;
    constexpr int kResultSize           = kWidth * kHeight;
    constexpr GLuint kInputValues[2][2] = {{200, 150}, {100, 50}};
    constexpr GLuint expectedValue_1    = 150;
    constexpr GLuint expectedValue_2    = 50;
    std::array<GLuint, kResultSize> outputValues;

    glBindTexture(GL_TEXTURE_3D, texture[0]);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_R32UI, kWidth, kHeight, kDepth);
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, kWidth, kHeight, kDepth, GL_RED_INTEGER,
                    GL_UNSIGNED_INT, kInputValues[0]);
    EXPECT_GL_NO_ERROR();

    glBindTexture(GL_TEXTURE_3D, texture[1]);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_R32UI, kWidth, kHeight, kDepth);
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, kWidth, kHeight, kDepth, GL_RED_INTEGER,
                    GL_UNSIGNED_INT, kInputValues[1]);
    EXPECT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    glUseProgram(program);

    glBindImageTexture(0, texture[0], 0, GL_FALSE, 1, GL_READ_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();
    glBindImageTexture(1, texture[1], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();
    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
    glUseProgram(0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
    glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture[1], 0, 0);
    glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, texture[1], 0, 1);
    EXPECT_GL_NO_ERROR();
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT, outputValues.data());
    EXPECT_GL_NO_ERROR();
    for (int i = 0; i < kResultSize; i++)
    {
        EXPECT_EQ(expectedValue_1, outputValues[i]);
    }
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    glReadPixels(0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT, outputValues.data());
    EXPECT_GL_NO_ERROR();
    for (int i = 0; i < kResultSize; i++)
    {
        EXPECT_EQ(expectedValue_2, outputValues[i]);
    }
}

// Use image uniform to read and write one layer of TextureCube in compute shader, and verify the
// contents.
TEST_P(ComputeShaderTest, BindImageTextureWithOneLayerTextureCube)
{
    // GL_FRAMEBUFFER_BARRIER_BIT is invalid on Nvidia Linux platform.
    // http://anglebug.com/42262394
    ANGLE_SKIP_TEST_IF(IsNVIDIA() && IsOpenGL() && IsLinux());

    GLTexture texture[2];
    GLFramebuffer framebuffer;
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(r32ui, binding = 0) readonly uniform highp uimage2D uImage_1;
layout(r32ui, binding = 1) writeonly uniform highp uimage2D uImage_2;
void main()
{
    uvec4 value = imageLoad(uImage_1, ivec2(gl_LocalInvocationID.xy));
    imageStore(uImage_2, ivec2(gl_LocalInvocationID.xy), value);
})";

    constexpr int kWidth = 1, kHeight = 1;
    constexpr int kResultSize           = kWidth * kHeight;
    constexpr GLuint kInputValues[2][1] = {{200}, {100}};
    constexpr GLuint expectedValue_1    = 200;
    constexpr GLuint expectedValue_2    = 100;
    std::array<GLuint, kResultSize> outputValues;

    glBindTexture(GL_TEXTURE_CUBE_MAP, texture[0]);
    glTexStorage2D(GL_TEXTURE_CUBE_MAP, 1, GL_R32UI, kWidth, kHeight);
    for (GLenum face = GL_TEXTURE_CUBE_MAP_POSITIVE_X; face <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z;
         face++)
    {
        glTexSubImage2D(face, 0, 0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT,
                        kInputValues[0]);
    }
    EXPECT_GL_NO_ERROR();

    glBindTexture(GL_TEXTURE_CUBE_MAP, texture[1]);
    glTexStorage2D(GL_TEXTURE_CUBE_MAP, 1, GL_R32UI, kWidth, kHeight);
    for (GLenum face = GL_TEXTURE_CUBE_MAP_POSITIVE_X; face <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z;
         face++)
    {
        glTexSubImage2D(face, 0, 0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT,
                        kInputValues[1]);
    }
    EXPECT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    glUseProgram(program);

    glBindImageTexture(0, texture[0], 0, GL_FALSE, 3, GL_READ_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();
    glBindImageTexture(1, texture[1], 0, GL_FALSE, 4, GL_WRITE_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();
    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
    glUseProgram(0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);

    for (GLenum face = 0; face < 6; face++)
    {
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, texture[1], 0);
        EXPECT_GL_NO_ERROR();
        glReadPixels(0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT, outputValues.data());
        EXPECT_GL_NO_ERROR();

        if (face == 4)
        {
            for (int i = 0; i < kResultSize; i++)
            {
                EXPECT_EQ(expectedValue_1, outputValues[i]);
            }
        }
        else
        {
            for (int i = 0; i < kResultSize; i++)
            {
                EXPECT_EQ(expectedValue_2, outputValues[i]);
            }
        }
    }
}

// Test to bind kinds of texture types, bind either the entire texture
// level or a single layer or face of the face level.
TEST_P(ComputeShaderTest, BindImageTextureWithMixTextureTypes)
{
    // GL_FRAMEBUFFER_BARRIER_BIT is invalid on Nvidia Linux platform.
    // http://anglebug.com/42262394
    ANGLE_SKIP_TEST_IF(IsNVIDIA() && IsOpenGL() && IsLinux());

    // http://anglebug.com/42263641
    ANGLE_SKIP_TEST_IF(IsIntel() && IsLinux() && IsOpenGL());

    GLTexture texture[4];
    GLFramebuffer framebuffer;
    const char csSource[] =
        R"(#version 310 es
        layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
        layout(r32ui, binding = 0) readonly uniform highp uimage2D uImage_1;
        layout(r32ui, binding = 1) readonly uniform highp uimage2D uImage_2;
        layout(r32ui, binding = 2) readonly uniform highp uimage3D uImage_3;
        layout(r32ui, binding = 3) writeonly uniform highp uimage2D uImage_4;
        void main()
        {
            uvec4 value_1 = imageLoad(uImage_1, ivec2(gl_LocalInvocationID.xy));
            uvec4 value_2 = imageLoad(uImage_2, ivec2(gl_LocalInvocationID.xy));
            uvec4 value_3 = imageLoad(uImage_3, ivec3(gl_LocalInvocationID.xyz));
            imageStore(uImage_4, ivec2(gl_LocalInvocationID.xy), value_1 + value_2 + value_3);
        })";

    constexpr int kWidth = 1, kHeight = 1, kDepth = 2;
    constexpr int kResultSize               = kWidth * kHeight;
    constexpr GLuint kInputValues2D[1]      = {11};
    constexpr GLuint KInputValues2DArray[2] = {23, 35};
    constexpr GLuint KInputValues3D[2]      = {102, 67};
    constexpr GLuint KInputValuesCube[1]    = {232};

    constexpr GLuint expectedValue_1 = 148;
    constexpr GLuint expectedValue_2 = 232;
    std::array<GLuint, kResultSize> outputValues;

    glBindTexture(GL_TEXTURE_2D, texture[0]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, kWidth, kHeight);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    kInputValues2D);
    EXPECT_GL_NO_ERROR();

    glBindTexture(GL_TEXTURE_2D_ARRAY, texture[1]);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_R32UI, kWidth, kHeight, kDepth);
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, kWidth, kHeight, kDepth, GL_RED_INTEGER,
                    GL_UNSIGNED_INT, KInputValues2DArray);
    EXPECT_GL_NO_ERROR();

    glBindTexture(GL_TEXTURE_3D, texture[2]);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_R32UI, kWidth, kHeight, kDepth);
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, kWidth, kHeight, kDepth, GL_RED_INTEGER,
                    GL_UNSIGNED_INT, KInputValues3D);
    EXPECT_GL_NO_ERROR();

    glBindTexture(GL_TEXTURE_CUBE_MAP, texture[3]);
    glTexStorage2D(GL_TEXTURE_CUBE_MAP, 1, GL_R32UI, kWidth, kHeight);
    for (GLenum face = GL_TEXTURE_CUBE_MAP_POSITIVE_X; face <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z;
         face++)
    {
        glTexSubImage2D(face, 0, 0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT,
                        KInputValuesCube);
    }
    EXPECT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(program, csSource);
    glUseProgram(program);

    glBindImageTexture(0, texture[0], 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();
    glBindImageTexture(1, texture[1], 0, GL_FALSE, 1, GL_READ_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();
    glBindImageTexture(2, texture[2], 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();
    glBindImageTexture(3, texture[3], 0, GL_FALSE, 4, GL_WRITE_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();
    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
    glUseProgram(0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);

    for (GLenum face = 0; face < 6; face++)
    {
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, texture[3], 0);
        EXPECT_GL_NO_ERROR();
        glReadPixels(0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT, outputValues.data());
        EXPECT_GL_NO_ERROR();

        if (face == 4)
        {
            for (int i = 0; i < kResultSize; i++)
            {
                EXPECT_EQ(expectedValue_1, outputValues[i]);
            }
        }
        else
        {
            for (int i = 0; i < kResultSize; i++)
            {
                EXPECT_EQ(expectedValue_2, outputValues[i]);
            }
        }
    }
}

// Verify an INVALID_OPERATION error is reported when querying GL_COMPUTE_WORK_GROUP_SIZE for a
// program which has not been linked successfully or which does not contain objects to form a
// compute shader.
TEST_P(ComputeShaderTest, QueryComputeWorkGroupSize)
{
    constexpr char kVS[] = R"(#version 310 es
void main()
{
})";

    constexpr char kFS[] = R"(#version 310 es
void main()
{
})";

    GLint workGroupSize[3];

    ANGLE_GL_PROGRAM(graphicsProgram, kVS, kFS);
    glGetProgramiv(graphicsProgram, GL_COMPUTE_WORK_GROUP_SIZE, workGroupSize);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    GLuint computeProgram = glCreateProgram();
    GLShader computeShader(GL_COMPUTE_SHADER);
    glAttachShader(computeProgram, computeShader);
    glLinkProgram(computeProgram);
    glDetachShader(computeProgram, computeShader);

    GLint linkStatus;
    glGetProgramiv(computeProgram, GL_LINK_STATUS, &linkStatus);
    ASSERT_GL_FALSE(linkStatus);

    glGetProgramiv(computeProgram, GL_COMPUTE_WORK_GROUP_SIZE, workGroupSize);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    glDeleteProgram(computeProgram);

    ASSERT_GL_NO_ERROR();
}

// Use groupMemoryBarrier and barrier to sync reads/writes order and the execution
// order of multiple shader invocations in compute shader.
TEST_P(ComputeShaderTest, GroupMemoryBarrierAndBarrierTest)
{
    // TODO(xinghua.cao@intel.com): Figure out why we get this error message
    // that shader uses features not recognized by this D3D version.
    ANGLE_SKIP_TEST_IF((IsAMD() || IsNVIDIA()) && IsD3D11());
    ANGLE_SKIP_TEST_IF(IsARM64() && IsWindows() && IsD3D());

    // http://anglebug.com/42263641
    ANGLE_SKIP_TEST_IF(IsIntel() && IsLinux() && IsOpenGL());

    GLTexture texture;
    GLFramebuffer framebuffer;

    // Each invocation first stores a single value in an image, then each invocation sums up
    // all the values in the image and stores the sum in the image. groupMemoryBarrier is
    // used to order reads/writes to variables stored in memory accessible to other shader
    // invocations, and barrier is used to control the relative execution order of multiple
    // shader invocations used to process a local work group.
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=2, local_size_y=2, local_size_z=1) in;
layout(r32i, binding = 0) uniform highp iimage2D image;
void main()
{
    uint x = gl_LocalInvocationID.x;
    uint y = gl_LocalInvocationID.y;
    imageStore(image, ivec2(gl_LocalInvocationID.xy), ivec4(x + y));
    groupMemoryBarrier();
    barrier();
    int sum = 0;
    for (int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 2; j++)
        {
            sum += imageLoad(image, ivec2(i, j)).x;
        }
    }
    groupMemoryBarrier();
    barrier();
    imageStore(image, ivec2(gl_LocalInvocationID.xy), ivec4(sum));
})";

    constexpr int kWidth = 2, kHeight = 2;
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32I, kWidth, kHeight);
    EXPECT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    glUseProgram(program);

    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32I);
    EXPECT_GL_NO_ERROR();

    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
    std::array<GLuint, kWidth * kHeight> outputValues;
    constexpr GLuint kExpectedValue = 4;
    glUseProgram(0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);

    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
    EXPECT_GL_NO_ERROR();
    glReadPixels(0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_INT, outputValues.data());
    EXPECT_GL_NO_ERROR();

    for (int i = 0; i < kWidth * kHeight; i++)
    {
        EXPECT_EQ(kExpectedValue, outputValues[i]);
    }
}

// Verify that a link error is generated when the sum of the number of active image uniforms and
// active shader storage blocks in a compute shader exceeds GL_MAX_COMBINED_SHADER_OUTPUT_RESOURCES.
TEST_P(ComputeShaderTest, ExceedCombinedShaderOutputResourcesInCS)
{
    GLint maxCombinedShaderOutputResources;
    GLint maxComputeShaderStorageBlocks;
    GLint maxComputeImageUniforms;

    glGetIntegerv(GL_MAX_COMBINED_SHADER_OUTPUT_RESOURCES, &maxCombinedShaderOutputResources);
    glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS, &maxComputeShaderStorageBlocks);
    glGetIntegerv(GL_MAX_COMPUTE_IMAGE_UNIFORMS, &maxComputeImageUniforms);

    ANGLE_SKIP_TEST_IF(maxCombinedShaderOutputResources >=
                       maxComputeShaderStorageBlocks + maxComputeImageUniforms);

    std::ostringstream computeShaderStream;
    computeShaderStream << "#version 310 es\n"
                           "layout(local_size_x = 3, local_size_y = 1, local_size_z = 1) in;\n"
                           "layout(shared, binding = 0) buffer blockName"
                           "{\n"
                           "    uint data;\n"
                           "} instance["
                        << maxComputeShaderStorageBlocks << "];\n";

    ASSERT_GE(maxComputeImageUniforms, 4);
    int numImagesInArray  = maxComputeImageUniforms / 2;
    int numImagesNonArray = maxComputeImageUniforms - numImagesInArray;
    for (int i = 0; i < numImagesNonArray; ++i)
    {
        computeShaderStream << "layout(r32f, binding = " << i << ") uniform highp image2D image"
                            << i << ";\n";
    }

    computeShaderStream << "layout(r32f, binding = " << numImagesNonArray
                        << ") uniform highp image2D imageArray[" << numImagesInArray << "];\n";

    computeShaderStream << "void main()\n"
                           "{\n"
                           "    uint val = 0u;\n"
                           "    vec4 val2 = vec4(0.0);\n";

    for (int i = 0; i < maxComputeShaderStorageBlocks; ++i)
    {
        computeShaderStream << "    val += instance[" << i << "].data; \n";
    }

    for (int i = 0; i < numImagesNonArray; ++i)
    {
        computeShaderStream << "    val2 += imageLoad(image" << i
                            << ", ivec2(gl_LocalInvocationID.xy)); \n";
    }

    for (int i = 0; i < numImagesInArray; ++i)
    {
        computeShaderStream << "    val2 += imageLoad(imageArray[" << i << "]"
                            << ", ivec2(gl_LocalInvocationID.xy)); \n";
    }

    computeShaderStream << "    instance[0].data = val + uint(val2.x);\n"
                           "}\n";

    GLuint computeProgram = CompileComputeProgram(computeShaderStream.str().c_str());
    EXPECT_EQ(0u, computeProgram);
}

// Tests running a compute shader writing to separate components on SSBO without explicit barrier.
// Based on dEQP test: KHR-GLES31.core.shader_storage_buffer_object.advanced-switchBuffers-cs
TEST_P(ComputeShaderTest, WriteToSeparateSSBOComponentsWithoutExplicitBarrier)
{
    constexpr char kCS[] = R"(#version 310 es
precision highp float;
precision highp int;

layout(local_size_x = 1) in;
layout(binding = 0, std430) buffer Input {
    uint cookie[4];
} g_in;

layout(binding = 1, std430) buffer Output {
    uvec4 digest;
};

void main()
{
    uint sum = g_in.cookie[0] + g_in.cookie[1] + g_in.cookie[2] + g_in.cookie[3];
    switch (sum)
    {
        case 0x000000FFu:
            digest.x = 0xFF000000u;
            break;
        case 0x0000FF00u:
            digest.y = 0x00FF0000u;
            break;
        case 0x00FF0000u:
            digest.z = 0x0000FF00u;
            break;
        case 0xFF000000u:
            digest.w = 0x000000FFu;
            break;
        default:
            break;
   }
})";
    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    EXPECT_GL_NO_ERROR();
    glUseProgram(program);

    std::array<GLBuffer, 4> ssboIn;
    GLBuffer ssboOut;

    const GLubyte data0[] = {0, 0, 0, 0x11, 0, 0, 0, 0x44, 0, 0, 0, 0x88, 0, 0, 0, 0x22};
    const GLubyte data1[] = {0, 0, 0x44, 0, 0, 0, 0x22, 0, 0, 0, 0x88, 0, 0, 0, 0x11, 0};
    const GLubyte data2[] = {0, 0x88, 0, 0, 0, 0x11, 0, 0, 0, 0x44, 0, 0, 0, 0x22, 0, 0};
    const GLubyte data3[] = {0x22, 0, 0, 0, 0x88, 0, 0, 0, 0x11, 0, 0, 0, 0x44, 0, 0, 0};

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboIn[0]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(data0), data0, GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboIn[1]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(data1), data1, GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboIn[2]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(data2), data2, GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboIn[3]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(data3), data3, GL_STATIC_DRAW);

    const GLubyte dataZero[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboOut);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(dataZero), dataZero, GL_STATIC_DRAW);

    // Since the write is expected to occur on separate components on the output SSBO, no explicit
    // barrier is used between the dispatch calls.
    for (uint32_t i = 0; i < 4; ++i)
    {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboIn[i]);
        glDispatchCompute(1, 1, 1);
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboOut);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    std::array<GLuint, 4> expectedOutput = {0xFF000000, 0x00FF0000, 0x0000FF00, 0x000000FF};
    auto actualOutput = MapBufferRange<const GLuint>(GL_SHADER_STORAGE_BUFFER, 0,
                                                     sizeof(expectedOutput), GL_MAP_READ_BIT);
    EXPECT_EQ(Span(expectedOutput), actualOutput);
}

// Test that uniform block with struct member in compute shader is supported.
TEST_P(ComputeShaderTest, UniformBlockWithStructMember)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=8) in;
layout(rgba8) uniform highp readonly image2D mImage2DInput;
layout(rgba8) uniform highp writeonly image2D mImage2DOutput;
struct S {
    ivec3 a;
    ivec2 b;
};

layout(std140, binding=0) uniform blockName {
    S bd;
} instanceName;
void main()
{
    ivec2 t1 = instanceName.bd.b;
    vec4 result2d = imageLoad(mImage2DInput, t1);
    imageStore(mImage2DOutput, ivec2(gl_LocalInvocationID.xy), result2d);
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    EXPECT_GL_NO_ERROR();
}

// Verify shared non-array variables can work correctly.
TEST_P(ComputeShaderTest, NonArraySharedVariable)
{
    // http://anglebug.com/42263641
    ANGLE_SKIP_TEST_IF(IsIntel() && IsLinux() && IsOpenGL());

    const char kCSShader[] = R"(#version 310 es
layout (local_size_x = 2, local_size_y = 2, local_size_z = 1) in;
layout (r32ui, binding = 0) readonly uniform highp uimage2D srcImage;
layout (r32ui, binding = 1) writeonly uniform highp uimage2D dstImage;
shared uint temp;
void main()
{
    if (gl_LocalInvocationID == uvec3(0, 0, 0))
    {
        temp = imageLoad(srcImage, ivec2(gl_LocalInvocationID.xy)).x;
    }
    groupMemoryBarrier();
    barrier();
    if (gl_LocalInvocationID == uvec3(1, 1, 0))
    {
        imageStore(dstImage, ivec2(gl_LocalInvocationID.xy), uvec4(temp));
    }
    else
    {
        uint inputValue = imageLoad(srcImage, ivec2(gl_LocalInvocationID.xy)).x;
        imageStore(dstImage, ivec2(gl_LocalInvocationID.xy), uvec4(inputValue));
    }
})";

    const std::array<GLuint, 4> inputData      = {{250, 200, 150, 100}};
    const std::array<GLuint, 4> expectedValues = {{250, 200, 150, 250}};
    runSharedMemoryTest<GLuint, 2, 2>(kCSShader, GL_R32UI, GL_UNSIGNED_INT, inputData,
                                      expectedValues);
}

// Verify shared non-struct array variables can work correctly.
TEST_P(ComputeShaderTest, NonStructArrayAsSharedVariable)
{
    // http://anglebug.com/42263641
    ANGLE_SKIP_TEST_IF(IsIntel() && IsLinux() && IsOpenGL());

    const char kCSShader[] = R"(#version 310 es
layout (local_size_x = 2, local_size_y = 2, local_size_z = 1) in;
layout (r32ui, binding = 0) readonly uniform highp uimage2D srcImage;
layout (r32ui, binding = 1) writeonly uniform highp uimage2D dstImage;
shared uint sharedData[2][2];
void main()
{
    uint inputData = imageLoad(srcImage, ivec2(gl_LocalInvocationID.xy)).x;
    sharedData[gl_LocalInvocationID.x][gl_LocalInvocationID.y] = inputData;
    groupMemoryBarrier();
    barrier();
    imageStore(dstImage, ivec2(gl_LocalInvocationID.xy),
                uvec4(sharedData[gl_LocalInvocationID.y][gl_LocalInvocationID.x]));
})";

    const std::array<GLuint, 4> inputData      = {{250, 200, 150, 100}};
    const std::array<GLuint, 4> expectedValues = {{250, 150, 200, 100}};
    runSharedMemoryTest<GLuint, 2, 2>(kCSShader, GL_R32UI, GL_UNSIGNED_INT, inputData,
                                      expectedValues);
}

// Verify shared struct array variables work correctly.
TEST_P(ComputeShaderTest, StructArrayAsSharedVariable)
{
    // http://anglebug.com/42263641
    ANGLE_SKIP_TEST_IF(IsIntel() && IsLinux() && IsOpenGL());

    const char kCSShader[] = R"(#version 310 es
layout (local_size_x = 2, local_size_y = 2, local_size_z = 1) in;
layout (r32ui, binding = 0) readonly uniform highp uimage2D srcImage;
layout (r32ui, binding = 1) writeonly uniform highp uimage2D dstImage;
struct SharedStruct
{
    uint data;
};
shared SharedStruct sharedData[2][2];
void main()
{
    uint inputData = imageLoad(srcImage, ivec2(gl_LocalInvocationID.xy)).x;
    sharedData[gl_LocalInvocationID.x][gl_LocalInvocationID.y].data = inputData;
    groupMemoryBarrier();
    barrier();
    imageStore(dstImage, ivec2(gl_LocalInvocationID.xy),
                uvec4(sharedData[gl_LocalInvocationID.y][gl_LocalInvocationID.x].data));
})";

    const std::array<GLuint, 4> inputData      = {{250, 200, 150, 100}};
    const std::array<GLuint, 4> expectedValues = {{250, 150, 200, 100}};
    runSharedMemoryTest<GLuint, 2, 2>(kCSShader, GL_R32UI, GL_UNSIGNED_INT, inputData,
                                      expectedValues);
}

// Verify using atomic functions without return value can work correctly.
TEST_P(ComputeShaderTest, AtomicFunctionsNoReturnValue)
{
    // http://anglebug.com/42263641
    ANGLE_SKIP_TEST_IF(IsIntel() && IsLinux() && IsOpenGL());

    // Fails to link on Android.  http://anglebug.com/42262519
    ANGLE_SKIP_TEST_IF(IsAndroid());

    ANGLE_SKIP_TEST_IF(IsARM64() && IsWindows() && IsD3D());

    const char kCSShader[] = R"(#version 310 es
layout (local_size_x = 8, local_size_y = 1, local_size_z = 1) in;
layout (r32ui, binding = 0) readonly uniform highp uimage2D srcImage;
layout (r32ui, binding = 1) writeonly uniform highp uimage2D dstImage;

const uint kSumIndex = 0u;
const uint kMinIndex = 1u;
const uint kMaxIndex = 2u;
const uint kOrIndex = 3u;
const uint kAndIndex = 4u;
const uint kXorIndex = 5u;
const uint kExchangeIndex = 6u;
const uint kCompSwapIndex = 7u;

shared highp uint results[8];

void main()
{
    if (gl_LocalInvocationID.x == kMinIndex || gl_LocalInvocationID.x == kAndIndex)
    {
        results[gl_LocalInvocationID.x] = 0xFFFFu;
    }
    else if (gl_LocalInvocationID.x == kCompSwapIndex)
    {
        results[gl_LocalInvocationID.x] = 1u;
    }
    else
    {
        results[gl_LocalInvocationID.x] = 0u;
    }
    memoryBarrierShared();
    barrier();

    uint value = imageLoad(srcImage, ivec2(gl_LocalInvocationID.xy)).x;
    atomicAdd(results[kSumIndex], value);
    atomicMin(results[kMinIndex], value);
    atomicMax(results[kMaxIndex], value);
    atomicOr(results[kOrIndex], value);
    atomicAnd(results[kAndIndex], value);
    atomicXor(results[kXorIndex], value);
    atomicExchange(results[kExchangeIndex], value);
    atomicCompSwap(results[kCompSwapIndex], value, 256u);
    memoryBarrierShared();
    barrier();

    imageStore(dstImage, ivec2(gl_LocalInvocationID.xy),
                uvec4(results[gl_LocalInvocationID.x]));
})";

    const std::array<GLuint, 8> inputData      = {{1, 2, 4, 8, 16, 32, 64, 128}};
    const std::array<GLuint, 8> expectedValues = {{255, 1, 128, 255, 0, 255, 128, 256}};
    runSharedMemoryTest<GLuint, 8, 1>(kCSShader, GL_R32UI, GL_UNSIGNED_INT, inputData,
                                      expectedValues);
}

// Verify using atomic functions in a non-initializer single assignment can work correctly.
TEST_P(ComputeShaderTest, AtomicFunctionsInNonInitializerSingleAssignment)
{
    // http://anglebug.com/42263641
    ANGLE_SKIP_TEST_IF(IsIntel() && IsLinux() && IsOpenGL());

    const char kCSShader[] = R"(#version 310 es
layout (local_size_x = 9, local_size_y = 1, local_size_z = 1) in;
layout (r32i, binding = 0) readonly uniform highp iimage2D srcImage;
layout (r32i, binding = 1) writeonly uniform highp iimage2D dstImage;

shared highp int sharedVariable;

shared highp int inputData[9];
shared highp int outputData[9];

void main()
{
    int inputValue = imageLoad(srcImage, ivec2(gl_LocalInvocationID.xy)).x;
    inputData[gl_LocalInvocationID.x] = inputValue;
    memoryBarrierShared();
    barrier();

    if (gl_LocalInvocationID.x == 0u)
    {
        sharedVariable = 0;

        outputData[0] = atomicAdd(sharedVariable, inputData[0]);
        outputData[1] = atomicMin(sharedVariable, inputData[1]);
        outputData[2] = atomicMax(sharedVariable, inputData[2]);
        outputData[3] = atomicAnd(sharedVariable, inputData[3]);
        outputData[4] = atomicOr(sharedVariable, inputData[4]);
        outputData[5] = atomicXor(sharedVariable, inputData[5]);
        outputData[6] = atomicExchange(sharedVariable, inputData[6]);
        outputData[7] = atomicCompSwap(sharedVariable, 64, inputData[7]);
        outputData[8] = atomicAdd(sharedVariable, inputData[8]);
    }
    memoryBarrierShared();
    barrier();

    imageStore(dstImage, ivec2(gl_LocalInvocationID.xy),
                ivec4(outputData[gl_LocalInvocationID.x]));
})";

    const std::array<GLint, 9> inputData      = {{1, 2, 4, 8, 16, 32, 64, 128, 1}};
    const std::array<GLint, 9> expectedValues = {{0, 1, 1, 4, 0, 16, 48, 64, 128}};
    runSharedMemoryTest<GLint, 9, 1>(kCSShader, GL_R32I, GL_INT, inputData, expectedValues);
}

// Verify using atomic functions in an initializers and using unsigned int works correctly.
TEST_P(ComputeShaderTest, AtomicFunctionsInitializerWithUnsigned)
{
    // http://anglebug.com/42263641
    ANGLE_SKIP_TEST_IF(IsIntel() && IsLinux() && IsOpenGL());

    constexpr char kCShader[] = R"(#version 310 es
layout (local_size_x = 9, local_size_y = 1, local_size_z = 1) in;
layout (r32ui, binding = 0) readonly uniform highp uimage2D srcImage;
layout (r32ui, binding = 1) writeonly uniform highp uimage2D dstImage;

shared highp uint sharedVariable;

shared highp uint inputData[9];
shared highp uint outputData[9];

void main()
{
    uint inputValue = imageLoad(srcImage, ivec2(gl_LocalInvocationID.xy)).x;
    inputData[gl_LocalInvocationID.x] = inputValue;
    memoryBarrierShared();
    barrier();

    if (gl_LocalInvocationID.x == 0u)
    {
        sharedVariable = 0u;

        uint addValue = atomicAdd(sharedVariable, inputData[0]);
        outputData[0] = addValue;
        uint minValue = atomicMin(sharedVariable, inputData[1]);
        outputData[1] = minValue;
        uint maxValue = atomicMax(sharedVariable, inputData[2]);
        outputData[2] = maxValue;
        uint andValue = atomicAnd(sharedVariable, inputData[3]);
        outputData[3] = andValue;
        uint orValue = atomicOr(sharedVariable, inputData[4]);
        outputData[4] = orValue;
        uint xorValue = atomicXor(sharedVariable, inputData[5]);
        outputData[5] = xorValue;
        uint exchangeValue = atomicExchange(sharedVariable, inputData[6]);
        outputData[6] = exchangeValue;
        uint compSwapValue = atomicCompSwap(sharedVariable, 64u, inputData[7]);
        outputData[7] = compSwapValue;
        uint sharedVariable = atomicAdd(sharedVariable, inputData[8]);
        outputData[8] = sharedVariable;

    }
    memoryBarrierShared();
    barrier();

    imageStore(dstImage, ivec2(gl_LocalInvocationID.xy),
                uvec4(outputData[gl_LocalInvocationID.x]));
})";

    constexpr std::array<GLuint, 9> kInputData      = {{1, 2, 4, 8, 16, 32, 64, 128, 1}};
    constexpr std::array<GLuint, 9> kExpectedValues = {{0, 1, 1, 4, 0, 16, 48, 64, 128}};
    runSharedMemoryTest<GLuint, 9, 1>(kCShader, GL_R32UI, GL_UNSIGNED_INT, kInputData,
                                      kExpectedValues);
}

// Verify using atomic functions inside expressions as unsigned int.
TEST_P(ComputeShaderTest, AtomicFunctionsReturnWithUnsigned)
{
    // http://anglebug.com/42263641
    ANGLE_SKIP_TEST_IF(IsIntel() && IsLinux() && IsOpenGL());

    constexpr char kCShader[] = R"(#version 310 es
layout (local_size_x = 9, local_size_y = 1, local_size_z = 1) in;
layout (r32ui, binding = 0) readonly uniform highp uimage2D srcImage;
layout (r32ui, binding = 1) writeonly uniform highp uimage2D dstImage;

shared highp uint sharedVariable;

shared highp uint inputData[9];
shared highp uint outputData[9];

void main()
{
    uint inputValue = imageLoad(srcImage, ivec2(gl_LocalInvocationID.xy)).x;
    inputData[gl_LocalInvocationID.x] = inputValue;
    memoryBarrierShared();
    barrier();

    if (gl_LocalInvocationID.x == 0u)
    {
        sharedVariable = 0u;

        outputData[0] = 1u + atomicAdd(sharedVariable, inputData[0]);
        outputData[1] = 1u + atomicMin(sharedVariable, inputData[1]);
        outputData[2] = 1u + atomicMax(sharedVariable, inputData[2]);
        outputData[3] = 1u + atomicAnd(sharedVariable, inputData[3]);
        outputData[4] = 1u + atomicOr(sharedVariable, inputData[4]);
        outputData[5] = 1u + atomicXor(sharedVariable, inputData[5]);
        outputData[6] = 1u + atomicExchange(sharedVariable, inputData[6]);
        outputData[7] = 1u + atomicCompSwap(sharedVariable, 64u, inputData[7]);
        outputData[8] = 1u + atomicAdd(sharedVariable, inputData[8]);
    }
    memoryBarrierShared();
    barrier();

    imageStore(dstImage, ivec2(gl_LocalInvocationID.xy),
                uvec4(outputData[gl_LocalInvocationID.x]));
})";

    constexpr std::array<GLuint, 9> kInputData      = {{1, 2, 4, 8, 16, 32, 64, 128, 1}};
    constexpr std::array<GLuint, 9> kExpectedValues = {{1, 2, 2, 5, 1, 17, 49, 65, 129}};
    runSharedMemoryTest<GLuint, 9, 1>(kCShader, GL_R32UI, GL_UNSIGNED_INT, kInputData,
                                      kExpectedValues);
}

// Verify that atomicOr() on mediump uint values with bit 15 set doesn't sign-extend the result.
// 0x8159 is within the mediump uint range [0, 2^16-1], so it must not become 0xFFFF8159.
TEST_P(ComputeShaderTest, AtomicOrMediumpUintHighBit)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(std430, binding = 0) buffer InOut
{
    mediump uint inputValues[4];
    mediump uint result;
    highp uint returnValue;
};

void main()
{
    for (int i = 0; i < 4; ++i)
    {
        atomicOr(result, inputValues[i]);
    }
    returnValue = atomicOr(result, inputValues[0]);
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    glUseProgram(program);

    constexpr std::array<GLuint, 6> kInitData = {{0x8000u, 0x0100u, 0x0058u, 0x0001u, 0u, 0u}};
    GLBuffer buffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(kInitData), kInitData.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer);

    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    EXPECT_GL_NO_ERROR();

    auto data = MapBufferRange<const GLuint>(GL_SHADER_STORAGE_BUFFER, 0, sizeof(kInitData),
                                             GL_MAP_READ_BIT);
    ASSERT_FALSE(data.empty());
    EXPECT_EQ(0x8159u, data[4]);
    EXPECT_EQ(0x8159u, data[5]);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    EXPECT_GL_NO_ERROR();
}

// Verify using nested atomic functions in expressions.
TEST_P(ComputeShaderTest, AtomicFunctionsReturnWithMultipleTypes)
{
    // http://anglebug.com/42263641
    ANGLE_SKIP_TEST_IF(IsIntel() && IsLinux() && IsOpenGL());

    constexpr char kCShader[] = R"(#version 310 es
layout (local_size_x = 4, local_size_y = 1, local_size_z = 1) in;
layout (r32ui, binding = 0) readonly uniform highp uimage2D srcImage;
layout (r32ui, binding = 1) writeonly uniform highp uimage2D dstImage;

shared highp uint sharedVariable;
shared highp int  indexVariable;

shared highp uint inputData[4];
shared highp uint outputData[4];

void main()
{
    uint inputValue = imageLoad(srcImage, ivec2(gl_LocalInvocationID.xy)).x;
    inputData[gl_LocalInvocationID.x] = inputValue;
    memoryBarrierShared();
    barrier();

    if (gl_LocalInvocationID.x == 0u)
    {
        sharedVariable = 0u;
        indexVariable = 2;

        outputData[0] = 1u + atomicAdd(sharedVariable, inputData[atomicAdd(indexVariable, -1)]);
        outputData[1] = 1u + atomicAdd(sharedVariable, inputData[atomicAdd(indexVariable, -1)]);
        outputData[2] = 1u + atomicAdd(sharedVariable, inputData[atomicAdd(indexVariable, -1)]);
        outputData[3] = atomicAdd(sharedVariable, 0u);

    }
    memoryBarrierShared();
    barrier();

    imageStore(dstImage, ivec2(gl_LocalInvocationID.xy),
                uvec4(outputData[gl_LocalInvocationID.x]));
})";

    constexpr std::array<GLuint, 4> kInputData      = {{1, 2, 3, 0}};
    constexpr std::array<GLuint, 4> kExpectedValues = {{1, 4, 6, 6}};
    runSharedMemoryTest<GLuint, 4, 1>(kCShader, GL_R32UI, GL_UNSIGNED_INT, kInputData,
                                      kExpectedValues);
}

// Basic uniform buffer functionality.
TEST_P(ComputeShaderTest, UniformBuffer)
{
    // http://anglebug.com/42263641
    ANGLE_SKIP_TEST_IF(IsIntel() && IsLinux() && IsOpenGL());

    GLTexture texture;
    GLBuffer buffer;
    GLFramebuffer framebuffer;
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
uniform uni
{
    uvec4 value;
};
layout(rgba32ui, binding = 0) writeonly uniform highp uimage2D uImage;
void main()
{
    imageStore(uImage, ivec2(gl_LocalInvocationID.xy), value);
})";

    constexpr int kWidth = 1, kHeight = 1;
    constexpr GLuint kInputValues[4] = {56, 57, 58, 59};

    glBindTexture(GL_TEXTURE_2D, texture);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32UI, kWidth, kHeight);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kWidth, kHeight, GL_RGBA_INTEGER, GL_UNSIGNED_INT,
                    kInputValues);
    EXPECT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    glUseProgram(program);

    GLint uniformBufferIndex = glGetUniformBlockIndex(program, "uni");
    EXPECT_NE(uniformBufferIndex, -1);
    std::array<GLuint, 4> data = {201, 202, 203, 204};
    glBindBuffer(GL_UNIFORM_BUFFER, buffer);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(GLuint) * 4, data.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, buffer);
    glUniformBlockBinding(program, uniformBufferIndex, 0);
    EXPECT_GL_NO_ERROR();

    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32UI);
    EXPECT_GL_NO_ERROR();

    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
    std::array<GLuint, kWidth * kHeight * 4> outputValues;
    glUseProgram(0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);

    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
    EXPECT_GL_NO_ERROR();
    glReadPixels(0, 0, kWidth, kHeight, GL_RGBA_INTEGER, GL_UNSIGNED_INT, outputValues.data());
    EXPECT_GL_NO_ERROR();

    for (int i = 0; i < kWidth * kHeight * 4; i++)
    {
        EXPECT_EQ(data[i], outputValues[i]);
    }
}

// Test that storing data to image and then loading the same image data works correctly.
TEST_P(ComputeShaderTest, StoreImageThenLoad)
{
    // http://anglebug.com/42263641
    ANGLE_SKIP_TEST_IF(IsIntel() && IsLinux() && IsOpenGL());

    const char kCSSource[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(r32ui, binding = 0) readonly uniform highp uimage2D uImage_1;
layout(r32ui, binding = 1) writeonly uniform highp uimage2D uImage_2;
void main()
{
    uvec4 value = imageLoad(uImage_1, ivec2(gl_LocalInvocationID.xy));
    imageStore(uImage_2, ivec2(gl_LocalInvocationID.xy), value);
})";

    constexpr GLuint kInputValues[3][1] = {{300}, {200}, {100}};
    GLTexture texture[3];
    glBindTexture(GL_TEXTURE_2D, texture[0]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, 1, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, kInputValues[0]);
    EXPECT_GL_NO_ERROR();

    glBindTexture(GL_TEXTURE_2D, texture[1]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, 1, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, kInputValues[1]);
    EXPECT_GL_NO_ERROR();

    glBindTexture(GL_TEXTURE_2D, texture[2]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, 1, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, kInputValues[2]);
    EXPECT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(program, kCSSource);
    glUseProgram(program);

    glBindImageTexture(0, texture[0], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI);
    glBindImageTexture(1, texture[1], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);

    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    EXPECT_GL_NO_ERROR();

    glBindImageTexture(0, texture[1], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI);
    glBindImageTexture(1, texture[2], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);

    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
    EXPECT_GL_NO_ERROR();

    GLuint outputValue;
    GLFramebuffer framebuffer;
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture[2], 0);
    glReadPixels(0, 0, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, &outputValue);
    EXPECT_GL_NO_ERROR();

    EXPECT_EQ(300u, outputValue);
}

// Test that loading image data and then storing data to the same image works correctly.
TEST_P(ComputeShaderTest, LoadImageThenStore)
{
    // http://anglebug.com/42263641
    ANGLE_SKIP_TEST_IF(IsIntel() && IsLinux() && IsOpenGL());

    const char kCSSource[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(r32ui, binding = 0) readonly uniform highp uimage2D uImage_1;
layout(r32ui, binding = 1) writeonly uniform highp uimage2D uImage_2;
void main()
{
    uvec4 value = imageLoad(uImage_1, ivec2(gl_LocalInvocationID.xy));
    imageStore(uImage_2, ivec2(gl_LocalInvocationID.xy), value);
})";

    constexpr GLuint kInputValues[3][1] = {{300}, {200}, {100}};
    GLTexture texture[3];
    glBindTexture(GL_TEXTURE_2D, texture[0]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, 1, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, kInputValues[0]);
    EXPECT_GL_NO_ERROR();

    glBindTexture(GL_TEXTURE_2D, texture[1]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, 1, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, kInputValues[1]);
    EXPECT_GL_NO_ERROR();

    glBindTexture(GL_TEXTURE_2D, texture[2]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, 1, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, kInputValues[2]);
    EXPECT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(program, kCSSource);
    glUseProgram(program);

    glBindImageTexture(0, texture[0], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI);
    glBindImageTexture(1, texture[1], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);

    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    EXPECT_GL_NO_ERROR();

    glBindImageTexture(0, texture[2], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI);
    glBindImageTexture(1, texture[0], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);

    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
    EXPECT_GL_NO_ERROR();

    GLuint outputValue;
    GLFramebuffer framebuffer;
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture[0], 0);
    glReadPixels(0, 0, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, &outputValue);
    EXPECT_GL_NO_ERROR();

    EXPECT_EQ(100u, outputValue);
}

// Test that the length of a struct buffer variable is supported.
TEST_P(ComputeShaderTest, ShaderStorageBlocksStructLength)
{
    const char kCSSource[] = R"(#version 310 es
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct Particle
{
    int len;
};

layout(binding = 0, std430) readonly buffer Buf1
{
    Particle particlesRead[];
};

layout(binding = 1, std430) buffer Buf2
{
    Particle particlesWrite[];
};

void main()
{
    int index = int(gl_GlobalInvocationID.x);
    particlesWrite[index].len = particlesRead.length();
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCSSource);
    EXPECT_GL_NO_ERROR();
}

// Test that scalar buffer variables are supported.
TEST_P(ComputeShaderTest, ShaderStorageBlocksScalar)
{
    const char kCSSource[] = R"(#version 310 es
layout(local_size_x=1) in;
layout(std140, binding = 0) buffer blockA {
    uvec3 uv;
    float f;
} instanceA;
layout(std140, binding = 1) buffer blockB {
    vec2 v;
    uint u[3];
    float f;
};
void main()
{
    f = instanceA.f;
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCSSource);
    EXPECT_GL_NO_ERROR();
}

// Test that vector buffer variables are supported.
TEST_P(ComputeShaderTest, ShaderStorageBlocksVector)
{
    const char kCSSource[] = R"(#version 310 es
layout(local_size_x=1) in;
layout(std140, binding = 0) buffer blockA {
    vec2 f;
} instanceA;
layout(std140, binding = 1) buffer blockB {
    vec3 f;
};
void main()
{
    f[1] = instanceA.f[0];
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCSSource);
    EXPECT_GL_NO_ERROR();
}

// Test that matrix buffer variables are supported.
TEST_P(ComputeShaderTest, ShaderStorageBlocksMatrix)
{
    const char kCSSource[] = R"(#version 310 es
layout(local_size_x=1) in;
layout(std140, binding = 0) buffer blockA {
    mat3x4 m;
} instanceA;
layout(std140, binding = 1) buffer blockB {
    mat3x4 m;
};
void main()
{
    m[0][1] = instanceA.m[0][1];
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCSSource);
    EXPECT_GL_NO_ERROR();
}

// Test that scalar array buffer variables are supported.
TEST_P(ComputeShaderTest, ShaderStorageBlocksScalarArray)
{
    const char kCSSource[] = R"(#version 310 es
layout(local_size_x=8) in;
layout(std140, binding = 0) buffer blockA {
    float f[8];
} instanceA;
layout(std140, binding = 1) buffer blockB {
    float f[8];
};
void main()
{
    f[gl_LocalInvocationIndex] = instanceA.f[gl_LocalInvocationIndex];
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCSSource);
    EXPECT_GL_NO_ERROR();
}

// Test that vector array buffer variables are supported.
TEST_P(ComputeShaderTest, ShaderStorageBlocksVectorArray)
{
    const char kCSSource[] = R"(#version 310 es
layout(local_size_x=4) in;
layout(std140, binding = 0) buffer blockA {
    vec2 v[4];
} instanceA;
layout(std140, binding = 1) buffer blockB {
    vec4 v[4];
};
void main()
{
    v[0][gl_LocalInvocationIndex] = instanceA.v[gl_LocalInvocationIndex][1];
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCSSource);
    EXPECT_GL_NO_ERROR();
}

// Test that matrix array buffer variables are supported.
TEST_P(ComputeShaderTest, ShaderStorageBlocksMatrixArray)
{
    const char kCSSource[] = R"(#version 310 es
layout(local_size_x=8) in;
layout(std140, binding = 0) buffer blockA {
    float v1[5];
    mat4 m[8];
} instanceA;
layout(std140, binding = 1) buffer blockB {
    vec2 v1[3];
    mat4 m[8];
};
void main()
{
    float data = instanceA.m[gl_LocalInvocationIndex][0][0];
    m[gl_LocalInvocationIndex][gl_LocalInvocationIndex][gl_LocalInvocationIndex] = data;
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCSSource);
    EXPECT_GL_NO_ERROR();
}

// Test that shader storage blocks only in assignment right is supported.
TEST_P(ComputeShaderTest, ShaderStorageBlocksInAssignmentRight)
{
    const char kCSSource[] = R"(#version 310 es
layout(local_size_x=8) in;
layout(std140, binding = 0) buffer blockA {
    float data[8];
} instanceA;
layout(r32f, binding = 0) writeonly uniform highp image2D imageOut;

void main()
{
    float data = 1.0;
    data = instanceA.data[gl_LocalInvocationIndex];
    imageStore(imageOut, ivec2(gl_LocalInvocationID.xy), vec4(data));
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCSSource);
    EXPECT_GL_NO_ERROR();
}

// Test that shader storage blocks with unsized array are supported.
TEST_P(ComputeShaderTest, ShaderStorageBlocksWithUnsizedArray)
{
    const char kCSSource[] = R"(#version 310 es
layout(local_size_x=8) in;
layout(std140, binding = 0) buffer blockA {
    float v[];
} instanceA;
layout(std140, binding = 0) buffer blockB {
    float v[];
} instanceB[1];

void main()
{
    float data = instanceA.v[gl_LocalInvocationIndex];
    instanceB[0].v[gl_LocalInvocationIndex * 2u + 1u] = data;
}
)";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCSSource);
    EXPECT_GL_NO_ERROR();
}

// Test that EOpIndexDirect/EOpIndexIndirect/EOpIndexDirectStruct nodes in ssbo EOpIndexInDirect
// don't need to calculate the offset and should be translated by OutputHLSL directly.
TEST_P(ComputeShaderTest, IndexAndDotOperatorsInSSBOIndexIndirectOperator)
{
    constexpr char kComputeShaderSource[] = R"(#version 310 es
layout(local_size_x=1) in;
layout(std140, binding = 0) buffer blockA {
    float v[4];
};
layout(std140, binding = 1) buffer blockB {
    float v[4];
} instanceB[1];
struct S
{
    uvec4 index[2];
} s;
void main()
{
        s.index[0] = uvec4(0u, 1u, 2u, 3u);
    float data = v[s.index[0].y];
    instanceB[0].v[s.index[0].x] = data;
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kComputeShaderSource);
    EXPECT_GL_NO_ERROR();
}

// Test that swizzle node in non-SSBO symbol works well.
TEST_P(ComputeShaderTest, ShaderStorageBlocksWithNonSSBOSwizzle)
{
    constexpr char kComputeShaderSource[] = R"(#version 310 es
layout(local_size_x=8) in;
layout(std140, binding = 0) buffer blockA {
    float v[8];
};
layout(std140, binding = 1) buffer blockB {
    float v[8];
} instanceB[1];

void main()
{
    float data = v[gl_GlobalInvocationID.x];
    instanceB[0].v[gl_GlobalInvocationID.x] = data;
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kComputeShaderSource);
    EXPECT_GL_NO_ERROR();
}

// Test that swizzle node in SSBO symbol works well.
TEST_P(ComputeShaderTest, ShaderStorageBlocksWithSSBOSwizzle)
{
    constexpr char kComputeShaderSource[] = R"(#version 310 es
layout(local_size_x=1) in;
layout(std140, binding = 0) buffer blockA {
    vec2 v;
};
layout(std140, binding = 1) buffer blockB {
    float v;
} instanceB[1];

void main()
{
    instanceB[0].v = v.x;
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kComputeShaderSource);
    EXPECT_GL_NO_ERROR();
}

// Test that a large struct array in std140 uniform block won't consume too much time.
TEST_P(ComputeShaderTest, LargeStructArraySize)
{
    constexpr char kComputeShaderSource[] = R"(#version 310 es
layout(local_size_x=8) in;
precision mediump float;

struct InstancingData
{
    mat4 transformation;
};

#define MAX_INSTANCE_COUNT 800

layout(std140) uniform InstanceBlock
{
    InstancingData instances[MAX_INSTANCE_COUNT];
};

layout(std140, binding = 1) buffer blockB {
    mat4 v[];
} instanceB;

void main()
{
    instanceB.v[gl_GlobalInvocationID.x] = instances[gl_GlobalInvocationID.x].transformation;
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kComputeShaderSource);
    EXPECT_GL_NO_ERROR();
}

// Check that it is not possible to create a compute shader when the context does not support ES
// 3.10
TEST_P(ComputeShaderTestES3, NotSupported)
{
    // Allow the system EGL to skip the test if the context version is not exactly 3.0, which is
    // possible when the driver does not support "EGL_ANGLE_create_context_backwards_compatible".
    ANGLE_SKIP_TEST_IF(isAtLeastClientVersion(3, 1));

    GLuint computeShaderHandle = glCreateShader(GL_COMPUTE_SHADER);
    EXPECT_EQ(0u, computeShaderHandle);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
}

// Test uniform dirty in compute shader, and verify the contents.
TEST_P(ComputeShaderTest, UniformDirty)
{
    // http://anglebug.com/42263641
    ANGLE_SKIP_TEST_IF(IsIntel() && IsLinux() && IsOpenGL());

    GLTexture texture[2];
    GLFramebuffer framebuffer;
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(r32ui, binding = 0) readonly uniform highp uimage2D uImage_1;
layout(r32ui, binding = 1) writeonly uniform highp uimage2D uImage_2;
uniform uint factor;
void main()
{
    uvec4 value = imageLoad(uImage_1, ivec2(gl_LocalInvocationID.xy));
    imageStore(uImage_2, ivec2(gl_LocalInvocationID.xy), value * factor);
})";

    constexpr int kWidth = 1, kHeight = 1;
    constexpr GLuint kInputValues[2][1] = {{200}, {100}};

    glBindTexture(GL_TEXTURE_2D, texture[0]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, kWidth, kHeight);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    kInputValues[0]);
    EXPECT_GL_NO_ERROR();

    glBindTexture(GL_TEXTURE_2D, texture[1]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, kWidth, kHeight);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    kInputValues[1]);
    EXPECT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    glUseProgram(program);

    glBindImageTexture(0, texture[0], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();

    glBindImageTexture(1, texture[1], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
    EXPECT_GL_NO_ERROR();

    glUniform1ui(glGetUniformLocation(program, "factor"), 2);
    EXPECT_GL_NO_ERROR();

    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    glUniform1ui(glGetUniformLocation(program, "factor"), 3);
    EXPECT_GL_NO_ERROR();

    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
    std::array<GLuint, kWidth * kHeight> outputValues;
    GLuint expectedValue = 600;
    glUseProgram(0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);

    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture[1], 0);
    EXPECT_GL_NO_ERROR();
    glReadPixels(0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT, outputValues.data());
    EXPECT_GL_NO_ERROR();

    for (int i = 0; i < kWidth * kHeight; i++)
    {
        EXPECT_EQ(expectedValue, outputValues[i]) << " index " << i;
    }
}

// Test storage buffer bound is unchanged, shader writes it, buffer content should be updated.
TEST_P(ComputeShaderTest, StorageBufferBoundUnchanged)
{
    // http://anglebug.com/40096654
    ANGLE_SKIP_TEST_IF(isSwiftshader());
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=16, local_size_y=16) in;
precision highp usampler2D;
uniform usampler2D tex;
uniform uint factor;
layout(std140, binding = 0) buffer buf {
    uint outData[16][16];
};

void main()
{
    uint x = gl_LocalInvocationID.x;
    uint y = gl_LocalInvocationID.y;
    float xCoord = float(x) / float(16);
    float yCoord = float(y) / float(16);
    outData[y][x] = texture(tex, vec2(xCoord, yCoord)).x + factor;
})";

    constexpr unsigned int kWidth  = 16;
    constexpr unsigned int kHeight = 16;
    GLTexture tex;
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, kWidth, kHeight);
    std::array<std::array<GLuint, kWidth>, kHeight> texels = {};
    for (unsigned int y = 0; y < kHeight; ++y)
    {
        for (unsigned int x = 0; x < kWidth; ++x)
        {
            texels[y][x] = x + y * kWidth;
        }
    }
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kWidth, kHeight, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    texels.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    // The array stride are rounded up to the base alignment of a vec4 for std140 layout.
    constexpr unsigned int kArrayStride = 16;
    GLBuffer ssbo;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, kWidth * kHeight * kArrayStride, nullptr,
                 GL_STREAM_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    EXPECT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    glUseProgram(program);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glUniform1i(glGetUniformLocation(program, "tex"), 0);
    glUniform1ui(glGetUniformLocation(program, "factor"), 2);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);

    glDispatchCompute(1, 1, 1);

    auto mapped1 = MapBufferRange<const GLuint>(GL_SHADER_STORAGE_BUFFER, 0,
                                                kWidth * kHeight * kArrayStride, GL_MAP_READ_BIT);
    EXPECT_GL_NO_ERROR();
    ASSERT_FALSE(mapped1.empty());
    for (unsigned int idx = 0; idx < kWidth * kHeight; idx++)
    {
        EXPECT_EQ(idx + 2, mapped1[idx * kArrayStride / sizeof(GLuint)]);
    }
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glUniform1ui(glGetUniformLocation(program, "factor"), 3);
    glDispatchCompute(1, 1, 1);

    auto mapped2 = MapBufferRange<const GLuint>(GL_SHADER_STORAGE_BUFFER, 0,
                                                kWidth * kHeight * kArrayStride, GL_MAP_READ_BIT);
    EXPECT_GL_NO_ERROR();
    ASSERT_FALSE(mapped2.empty());
    for (unsigned int idx = 0; idx < kWidth * kHeight; idx++)
    {
        EXPECT_EQ(idx + 3, mapped2[idx * kArrayStride / sizeof(GLuint)]);
    }
}

// Test imageSize to access mipmap slice.
TEST_P(ComputeShaderTest, ImageSizeMipmapSlice)
{
    // TODO(xinghua.cao@intel.com): http://anglebug.com/42261780
    ANGLE_SKIP_TEST_IF(IsIntel() && IsLinux());

    // http://anglebug.com/42263018
    ANGLE_SKIP_TEST_IF(IsWindows() && IsNVIDIA() && IsD3D11());

    GLTexture texture[2];
    GLFramebuffer framebuffer;
    const char kCS[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(r32ui, binding = 0) readonly uniform highp uimage2D uImage_1;
layout(rgba32ui, binding = 1) writeonly uniform highp uimage2D uImage_2;
void main()
{
    ivec2 size = imageSize(uImage_1);
    imageStore(uImage_2, ivec2(gl_LocalInvocationID.xy), uvec4(size, 0, 0));
})";

    constexpr int kWidth1 = 8, kHeight1 = 4, kWidth2 = 1, kHeight2 = 1;
    constexpr GLuint kInputValues[] = {0, 0, 0, 0};

    glBindTexture(GL_TEXTURE_2D, texture[0]);
    glTexStorage2D(GL_TEXTURE_2D, 2, GL_R32UI, kWidth1, kHeight1);
    EXPECT_GL_NO_ERROR();

    glBindTexture(GL_TEXTURE_2D, texture[1]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32UI, kWidth2, kHeight2);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kWidth2, kHeight2, GL_RGBA_INTEGER, GL_UNSIGNED_INT,
                    kInputValues);
    EXPECT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    glUseProgram(program);

    glBindImageTexture(0, texture[0], 1, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI);
    glBindImageTexture(1, texture[1], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32UI);

    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
    std::array<GLuint, kWidth2 * kHeight2 * 4> outputValues;
    constexpr std::array<GLuint, 2> expectedValue = {4, 2};
    glUseProgram(0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);

    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture[1], 0);
    EXPECT_GL_NO_ERROR();
    glReadPixels(0, 0, kWidth2, kHeight2, GL_RGBA_INTEGER, GL_UNSIGNED_INT, outputValues.data());
    EXPECT_GL_NO_ERROR();

    for (int i = 0; i < kWidth2 * kHeight2; i++)
    {
        EXPECT_EQ(expectedValue[i], outputValues[i]);
        EXPECT_EQ(expectedValue[i + 1], outputValues[i + 1]);
    }
}

// Test imageLoad to access mipmap slice.
TEST_P(ComputeShaderTest, ImageLoadMipmapSlice)
{
    // TODO(xinghua.cao@intel.com): http://anglebug.com/42261780
    ANGLE_SKIP_TEST_IF(IsIntel() && IsLinux());

    GLTexture texture[2];
    GLFramebuffer framebuffer;
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(r32ui, binding = 0) readonly uniform highp uimage2D uImage_1;
layout(r32ui, binding = 1) writeonly uniform highp uimage2D uImage_2;
void main()
{
    uvec4 value = imageLoad(uImage_1, ivec2(gl_LocalInvocationID.xy));
    imageStore(uImage_2, ivec2(gl_LocalInvocationID.xy), value);
})";

    constexpr int kWidth1 = 2, kHeight1 = 2, kWidth2 = 1, kHeight2 = 1;
    constexpr GLuint kInputValues11[] = {3, 3, 3, 3};
    constexpr GLuint kInputValues12[] = {2};
    constexpr GLuint kInputValues2[]  = {1};

    glBindTexture(GL_TEXTURE_2D, texture[0]);
    glTexStorage2D(GL_TEXTURE_2D, 2, GL_R32UI, kWidth1, kHeight1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kWidth1, kHeight1, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    kInputValues11);
    glTexSubImage2D(GL_TEXTURE_2D, 1, 0, 0, kWidth2, kHeight2, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    kInputValues12);
    EXPECT_GL_NO_ERROR();

    glBindTexture(GL_TEXTURE_2D, texture[1]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, kWidth2, kHeight2);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kWidth2, kHeight2, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    kInputValues2);
    EXPECT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    glUseProgram(program);

    glBindImageTexture(0, texture[0], 1, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI);
    glBindImageTexture(1, texture[1], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);

    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
    GLuint outputValues;
    constexpr GLuint expectedValue = 2;
    glUseProgram(0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);

    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture[1], 0);
    EXPECT_GL_NO_ERROR();
    glReadPixels(0, 0, kWidth2, kHeight2, GL_RED_INTEGER, GL_UNSIGNED_INT, &outputValues);
    EXPECT_GL_NO_ERROR();
    EXPECT_EQ(expectedValue, outputValues);
}

// Test imageStore to access mipmap slice.
TEST_P(ComputeShaderTest, ImageStoreMipmapSlice)
{
    // TODO(xinghua.cao@intel.com): http://anglebug.com/42261780
    ANGLE_SKIP_TEST_IF(IsIntel() && IsLinux() && IsOpenGL());

    GLTexture texture[2];
    GLFramebuffer framebuffer;
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(r32ui, binding = 0) readonly uniform highp uimage2D uImage_1;
layout(r32ui, binding = 1) writeonly uniform highp uimage2D uImage_2;
void main()
{
    uvec4 value = imageLoad(uImage_1, ivec2(gl_LocalInvocationID.xy));
    imageStore(uImage_2, ivec2(gl_LocalInvocationID.xy), value);
})";

    constexpr int kWidth1 = 1, kHeight1 = 1, kWidth2 = 2, kHeight2 = 2;
    constexpr GLuint kInputValues1[]  = {3};
    constexpr GLuint kInputValues21[] = {2, 2, 2, 2};
    constexpr GLuint kInputValues22[] = {1};

    glBindTexture(GL_TEXTURE_2D, texture[0]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, kWidth1, kHeight1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kWidth1, kHeight1, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    kInputValues1);
    EXPECT_GL_NO_ERROR();

    glBindTexture(GL_TEXTURE_2D, texture[1]);
    glTexStorage2D(GL_TEXTURE_2D, 2, GL_R32UI, kWidth2, kHeight2);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kWidth2, kHeight2, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    kInputValues21);
    glTexSubImage2D(GL_TEXTURE_2D, 1, 0, 0, kWidth1, kHeight1, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    kInputValues22);
    EXPECT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    glUseProgram(program);

    glBindImageTexture(0, texture[0], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI);
    glBindImageTexture(1, texture[1], 1, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);

    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
    GLuint outputValues;
    constexpr GLuint expectedValue = 3;
    glUseProgram(0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);

    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture[1], 1);
    EXPECT_GL_NO_ERROR();
    glReadPixels(0, 0, kWidth1, kHeight1, GL_RED_INTEGER, GL_UNSIGNED_INT, &outputValues);
    EXPECT_GL_NO_ERROR();
    EXPECT_EQ(expectedValue, outputValues);
}

// Test that a resource is bound on render pipeline output, and then it's bound as the compute
// pipeline input. It works well. See http://anglebug.com/42262319
TEST_P(ComputeShaderTest, DrawTexture1DispatchTexture2)
{
    // http://anglebug.com/42263641
    ANGLE_SKIP_TEST_IF(IsIntel() && IsLinux() && IsOpenGL());

    ANGLE_SKIP_TEST_IF(!EnsureGLExtensionEnabled("GL_EXT_color_buffer_float"));

    const char kCSSource[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
precision highp sampler2D;
uniform sampler2D tex;
layout(rgba32f, binding = 0) writeonly uniform highp image2D image;
void main()
{
    vec4 value = texelFetch(tex, ivec2(gl_LocalInvocationID.xy), 0);
    imageStore(image, ivec2(gl_LocalInvocationID.xy), vec4(value.x - 1.0, 1.0, 0.0, value.w - 1.0));
})";

    const char kVSSource[] = R"(#version 310 es
layout (location = 0) in vec2 pos;
out vec2 texCoord;
void main(void) {
    texCoord = 0.5*pos + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
})";

    const char kFSSource[] = R"(#version 310 es
precision highp float;
uniform sampler2D tex;
in vec2 texCoord;
out vec4 fragColor;
void main(void) {
    fragColor = texture(tex, texCoord);
})";

    GLuint aPosLoc = 0;
    ANGLE_GL_PROGRAM(program, kVSSource, kFSSource);
    ANGLE_GL_COMPUTE_PROGRAM(csProgram, kCSSource);
    glBindAttribLocation(program, aPosLoc, "pos");
    GLBuffer buffer;
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    GLfloat vertices[] = {-1, -1, 1, -1, -1, 1, 1, 1};
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 8, vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(aPosLoc, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(aPosLoc);

    constexpr GLfloat kInputValues[4] = {1.0, 0.0, 0.0, 1.0};
    constexpr GLfloat kZero[4]        = {0.0, 0.0, 0.0, 0.0};
    GLFramebuffer framebuffer;
    GLTexture texture[3];
    glBindTexture(GL_TEXTURE_2D, texture[0]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, 1, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RGBA, GL_FLOAT, kInputValues);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, texture[1]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, 1, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RGBA, GL_FLOAT, kZero);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, texture[2]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, 1, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RGBA, GL_FLOAT, kZero);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glUseProgram(program);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture[0]);
    glUniform1i(glGetUniformLocation(program, "tex"), 0);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture[1], 0);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_GL_NO_ERROR();
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
    GLfloat actual[4];
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, actual);
    EXPECT_GL_NO_ERROR();
    EXPECT_EQ(1.0, actual[0]);
    EXPECT_EQ(0.0, actual[1]);
    EXPECT_EQ(0.0, actual[2]);
    EXPECT_EQ(1.0, actual[3]);

    glUseProgram(csProgram);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture[1]);
    glUniform1i(glGetUniformLocation(program, "tex"), 0);
    glBindImageTexture(0, texture[2], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture[2], 0);
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, actual);
    EXPECT_GL_NO_ERROR();
    EXPECT_EQ(0.0, actual[0]);
    EXPECT_EQ(1.0, actual[1]);
    EXPECT_EQ(0.0, actual[2]);
    EXPECT_EQ(0.0, actual[3]);
}

// Test that render pipeline and compute pipeline access to the same texture.
// Steps:
//   1. DispatchCompute.
//   2. DrawArrays.
TEST_P(ComputeShaderTest, DispatchDraw)
{
    // http://anglebug.com/42263641
    ANGLE_SKIP_TEST_IF(IsIntel() && IsLinux() && IsOpenGL());

    const char kCSSource[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(rgba32f, binding = 0) writeonly uniform highp image2D image;
void main()
{
    imageStore(image, ivec2(gl_LocalInvocationID.xy), vec4(0.0, 0.0, 1.0, 1.0));
})";

    const char kVSSource[] = R"(#version 310 es
layout (location = 0) in vec2 pos;
out vec2 texCoord;
void main(void) {
    texCoord = 0.5*pos + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
})";

    const char kFSSource[] = R"(#version 310 es
precision highp float;
uniform sampler2D tex;
in vec2 texCoord;
out vec4 fragColor;
void main(void) {
    fragColor = texture(tex, texCoord);
})";

    GLuint aPosLoc = 0;
    ANGLE_GL_PROGRAM(program, kVSSource, kFSSource);
    glBindAttribLocation(program, aPosLoc, "pos");
    GLBuffer buffer;
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    GLfloat vertices[] = {-1, -1, 1, -1, -1, 1, 1, 1};
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 8, vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(aPosLoc, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(aPosLoc);

    constexpr GLfloat kInputValues[4] = {1.0, 0.0, 0.0, 1.0};
    GLTexture texture;
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, 1, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RGBA, GL_FLOAT, kInputValues);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    EXPECT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(csProgram, kCSSource);
    glUseProgram(csProgram);

    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
    glUseProgram(program);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_PIXEL_COLOR_EQ(1, 1, GLColor::blue);
}

// Test that render pipeline and compute pipeline access to the same texture.
// Steps:
//   1. DrawArrays.
//   2. DispatchCompute.
//   3. DispatchCompute.
//   4. DrawArrays.
TEST_P(ComputeShaderTest, DrawDispatchDispatchDraw)
{
    // http://anglebug.com/42263641
    ANGLE_SKIP_TEST_IF(IsIntel() && IsLinux() && IsOpenGL());

    const char kCSSource[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(rgba32f, binding = 0) writeonly uniform highp image2D image;
uniform float factor;
void main()
{
    imageStore(image, ivec2(gl_LocalInvocationID.xy), vec4(factor, 0.0, 1.0, 1.0));
})";

    const char kVSSource[] = R"(#version 310 es
layout (location = 0) in vec2 pos;
out vec2 texCoord;
void main(void) {
    texCoord = 0.5*pos + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
})";

    const char kFSSource[] = R"(#version 310 es
precision highp float;
uniform sampler2D tex;
in vec2 texCoord;
out vec4 fragColor;
void main(void) {
    fragColor = texture(tex, texCoord);
})";

    GLuint aPosLoc = 0;
    ANGLE_GL_PROGRAM(program, kVSSource, kFSSource);
    glBindAttribLocation(program, aPosLoc, "pos");
    GLBuffer buffer;
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    GLfloat vertices[] = {-1, -1, 1, -1, -1, 1, 1, 1};
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 8, vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(aPosLoc, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(aPosLoc);

    constexpr GLfloat kInputValues[4] = {1.0, 0.0, 0.0, 1.0};
    GLTexture texture;
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, 1, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RGBA, GL_FLOAT, kInputValues);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glUseProgram(program);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(csProgram, kCSSource);
    glUseProgram(csProgram);
    glUniform1f(glGetUniformLocation(csProgram, "factor"), 0.0);
    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    EXPECT_GL_NO_ERROR();

    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    EXPECT_GL_NO_ERROR();

    glUniform1f(glGetUniformLocation(csProgram, "factor"), 1.0);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
    EXPECT_GL_NO_ERROR();

    glUseProgram(program);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_PIXEL_COLOR_EQ(1, 1, GLColor::magenta);
}

// Test that render pipeline and compute pipeline access to the same texture.
// Steps:
//   1. DispatchCompute.
//   2. DrawArrays.
//   3. DrawArrays.
//   4. DispatchCompute.
TEST_P(ComputeShaderTest, DispatchDrawDrawDispatch)
{
    // http://anglebug.com/42263641
    ANGLE_SKIP_TEST_IF(IsIntel() && IsLinux() && IsOpenGL());

    const char kCSSource[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(rgba32f, binding = 0) writeonly uniform highp image2D image;

void main()
{
    imageStore(image, ivec2(gl_LocalInvocationID.xy), vec4(0.0, 0.0, 1.0, 1.0));
})";

    const char kVSSource[] = R"(#version 310 es
layout (location = 0) in vec2 pos;
out vec2 texCoord;
void main(void) {
    texCoord = 0.5*pos + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
})";

    const char kFSSource[] = R"(#version 310 es
precision highp float;
uniform sampler2D tex;
in vec2 texCoord;
uniform float factor;
out vec4 fragColor;
void main(void) {
    fragColor = texture(tex, texCoord) + vec4(factor, 0.0, 0.0, 0.0);
})";

    GLuint aPosLoc = 0;
    ANGLE_GL_PROGRAM(program, kVSSource, kFSSource);
    glBindAttribLocation(program, aPosLoc, "pos");
    GLBuffer buffer;
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    GLfloat vertices[] = {-1, -1, 1, -1, -1, 1, 1, 1};
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 8, vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(aPosLoc, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(aPosLoc);

    constexpr GLfloat kInputValues[4] = {1.0, 0.0, 0.0, 1.0};
    GLTexture texture;
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, 1, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RGBA, GL_FLOAT, kInputValues);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    ANGLE_GL_COMPUTE_PROGRAM(csProgram, kCSSource);
    glUseProgram(csProgram);
    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
    EXPECT_GL_NO_ERROR();

    glUseProgram(program);
    glUniform1f(glGetUniformLocation(program, "factor"), 0.0);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_GL_NO_ERROR();

    glUniform1f(glGetUniformLocation(program, "factor"), 1.0);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    glUseProgram(csProgram);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
    EXPECT_GL_NO_ERROR();

    glUseProgram(program);
    glUniform1f(glGetUniformLocation(program, "factor"), 0.0);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_PIXEL_COLOR_EQ(1, 1, GLColor::blue);
}

// Test color texture sample from fragment shader and then read access from compute
TEST_P(ComputeShaderTest, DrawReadDrawDispatch)
{
    const char kVS[] = R"(#version 310 es
layout (location = 0) in vec3 pos;
void main(void) {
    gl_Position = vec4(pos, 1.0);
})";

    const char kFS[] = R"(#version 310 es
precision highp float;
uniform sampler2D tex;
out vec4 fragColor;
void main(void) {
        fragColor = texture(tex,vec2(0,0));
})";

    // Create color texture
    GLTexture colorTexture;
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, colorTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 1, 1);
    const GLColor textureData = GLColor::green;
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, textureData.data());

    // Render to surface with texture sample
    ANGLE_GL_PROGRAM(graphicsProgram, kVS, kFS);
    glUseProgram(graphicsProgram);
    const auto &quadVertices = GetQuadVertices();
    GLBuffer arrayBuffer;
    glBindBuffer(GL_ARRAY_BUFFER, arrayBuffer);
    glBufferData(GL_ARRAY_BUFFER, quadVertices.size() * sizeof(Vector3), quadVertices.data(),
                 GL_STATIC_DRAW);
    GLint positionAttributeLocation = 0;
    glBindAttribLocation(graphicsProgram, positionAttributeLocation, "pos");
    glVertexAttribPointer(positionAttributeLocation, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(positionAttributeLocation);

    GLint uTextureLocation = glGetUniformLocation(graphicsProgram, "tex");
    ASSERT_NE(-1, uTextureLocation);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, colorTexture);
    glUniform1i(uTextureLocation, 0);
    // Sample the color texture from fragment shader and verify. This flushes out commands which
    // ensures there will be no layout transition in next renderPass.
    glDisable(GL_DEPTH_TEST);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    EXPECT_PIXEL_COLOR_EQ(0, 0, textureData);

    // Sample the texture from fragment shader. No image layout transition expected here.
    glDrawArrays(GL_TRIANGLES, 0, 6);

    // Sample it from compute shader while the renderPass also sample from the same texture
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(std140, binding=0) buffer buf {
    vec4 outData;
};
uniform sampler2D u_tex2D;
void main()
{
    outData = texture(u_tex2D, vec2(gl_LocalInvocationID.xy));
})";
    ANGLE_GL_COMPUTE_PROGRAM(computeProgram, kCS);
    glUseProgram(computeProgram);
    uTextureLocation = glGetUniformLocation(computeProgram, "u_tex2D");
    ASSERT_NE(-1, uTextureLocation);
    glUniform1i(uTextureLocation, 0);
    GLBuffer ssbo;
    const std::vector<GLfloat> initialData(4, 0.0f);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, Span(initialData).size_bytes(), initialData.data(),
                 GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    auto mapped = MapBufferRange<const GLfloat>(GL_SHADER_STORAGE_BUFFER, 0,
                                                Span(initialData).size_bytes(), GL_MAP_READ_BIT);
    ASSERT_FALSE(mapped.empty());
    angle::Vector4 expected = textureData.toNormalizedVector();
    EXPECT_NEAR(expected[0], mapped[0], 0.001);
    EXPECT_NEAR(expected[1], mapped[1], 0.001);
    EXPECT_NEAR(expected[2], mapped[2], 0.001);
    EXPECT_NEAR(expected[3], mapped[3], 0.001);
    EXPECT_GL_NO_ERROR();
}

// Test that invalid memory barrier will produce an error.
TEST_P(ComputeShaderTest, InvalidMemoryBarrier)
{
    GLbitfield barriers = 0;
    glMemoryBarrier(barriers);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
}

// test atomic counter increment
// http://anglebug.com/42261924
TEST_P(ComputeShaderTest, AtomicCounterIncrement)
{
    constexpr char kComputeShader[] = R"(#version 310 es
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(binding = 1, std430) buffer Output {
  uint preGet[1];
  uint increment[1];
  uint postGet[1];
} sb_in;
layout(binding=0) uniform atomic_uint counter0;

void main(void)
{
  uint id = (gl_GlobalInvocationID.x);
  sb_in.preGet[0u]    = atomicCounter(counter0);
  sb_in.increment[0u] = atomicCounterIncrement(counter0);
  sb_in.postGet[0u]   = atomicCounter(counter0);
}
)";
    ANGLE_GL_COMPUTE_PROGRAM(program, kComputeShader);
    EXPECT_GL_NO_ERROR();

    glUseProgram(program);

    constexpr unsigned int kBytesPerComponent = sizeof(GLuint);

    GLBuffer shaderStorageBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, shaderStorageBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 3 * kBytesPerComponent, nullptr, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, shaderStorageBuffer);
    EXPECT_GL_NO_ERROR();

    constexpr GLuint atomicBufferInitialData[] = {2u};
    GLuint atomicBuffer;
    glGenBuffers(1, &atomicBuffer);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, atomicBuffer);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(atomicBufferInitialData), atomicBufferInitialData,
                 GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, atomicBuffer);
    EXPECT_GL_NO_ERROR();

    glDispatchCompute(1, 1, 1);
    glFinish();
    EXPECT_GL_NO_ERROR();

    // read back
    auto mapped = MapBufferRange<const GLuint>(GL_SHADER_STORAGE_BUFFER, 0, 3 * kBytesPerComponent,
                                               GL_MAP_READ_BIT);
    ASSERT_FALSE(mapped.empty());
    EXPECT_EQ(2u, mapped[0]);
    EXPECT_EQ(2u, mapped[1]);
    EXPECT_EQ(3u, mapped[2]);

    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
}

// Create a 'very large' array inside of a function in a compute shader.
TEST_P(ComputeShaderTest, VeryLargeArrayInsideFunction)
{
    constexpr char kComputeShader[] = R"(#version 310 es
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(binding = 0, std430) buffer Output {
  int value[1];
} output_data;

void main()
{
    int values[1000];
    for (int i = 0; i < values.length(); i++)
    {
        values[i] = 0;
    }

    int total = 0;
    for (int i = 0; i < values.length(); i++)
    {
        total += i;
        values[i] = total;
    }
    output_data.value[0u] = values[1000-1];
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kComputeShader);
    EXPECT_GL_NO_ERROR();

    glUseProgram(program);

    constexpr unsigned int kBytesPerComponent = sizeof(GLint);

    GLBuffer shaderStorageBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, shaderStorageBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 1 * kBytesPerComponent, nullptr, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, shaderStorageBuffer);
    EXPECT_GL_NO_ERROR();

    glDispatchCompute(1, 1, 1);
    glFinish();
    EXPECT_GL_NO_ERROR();

    // read back
    auto mapped = MapBufferRange<const GLint>(GL_SHADER_STORAGE_BUFFER, 0, 1 * kBytesPerComponent,
                                              GL_MAP_READ_BIT);
    ASSERT_FALSE(mapped.empty());
    EXPECT_EQ(499500, mapped[0]);

    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
}

// Test that render pipeline and compute pipeline access to the same texture.
// Steps:
//   1. Clear the texture and DrawArrays.
//   2. DispatchCompute to set the image's first pixel to a specific color.
//   3. DrawArrays and check data.
TEST_P(ComputeShaderTest, DrawDispatchDrawPreserve)
{
    // http://anglebug.com/42263641
    ANGLE_SKIP_TEST_IF(IsIntel() && IsLinux() && IsOpenGL());

    const char kCSSource[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1) in;
layout(rgba8, binding = 0) writeonly uniform highp image2D image;
void main()
{
    imageStore(image, ivec2(0, 0), vec4(0.0, 0.0, 1.0, 1.0));
})";

    const char kVSSource[] = R"(#version 310 es
layout (location = 0) in vec2 pos;
in vec4 inTex;
out vec4 texCoord;
void main(void) {
    texCoord = inTex;
    gl_Position = vec4(pos, 0.0, 1.0);
})";

    const char kFSSource[] = R"(#version 310 es
precision highp float;
uniform sampler2D tex;
in vec4 texCoord;
out vec4 fragColor;
void main(void) {
    fragColor = texture(tex, texCoord.xy);
})";
    GLuint aPosLoc         = 0;
    ANGLE_GL_PROGRAM(program, kVSSource, kFSSource);
    glBindAttribLocation(program, aPosLoc, "pos");

    std::vector<uint8_t> data(4 * getWindowWidth() * getWindowHeight(), 0);
    for (int i = 0; i < getWindowWidth() * getWindowHeight(); i++)
    {
        data[i * 4]     = 0xff;
        data[i * 4 + 1] = 0;
        data[i * 4 + 2] = 0;
        data[i * 4 + 3] = 0xff;
    }
    GLTexture texture;
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexStorage2D(GL_TEXTURE_2D, 2, GL_RGBA8, getWindowWidth(), getWindowHeight());
    // Clear the texture level 0 to Red.
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, getWindowWidth(), getWindowHeight(), GL_RGBA,
                    GL_UNSIGNED_BYTE, data.data());
    for (int i = 0; i < getWindowWidth() * getWindowHeight(); i++)
    {
        data[i * 4]     = 0;
        data[i * 4 + 1] = 0xff;
        data[i * 4 + 2] = 0;
        data[i * 4 + 3] = 0xff;
    }
    // Clear the texture level 1 to Green.
    glTexSubImage2D(GL_TEXTURE_2D, 1, 0, 0, getWindowWidth() / 2, getWindowHeight() / 2, GL_RGBA,
                    GL_UNSIGNED_BYTE, data.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glUseProgram(program);
    GLfloat vertices[]  = {-1, -1, 1, -1, -1, 1, 1, 1};
    GLfloat texCoords[] = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    GLint pos           = glGetAttribLocation(program, "pos");
    glEnableVertexAttribArray(pos);
    glVertexAttribPointer(pos, 2, GL_FLOAT, GL_FALSE, 0, vertices);
    GLint posTex = glGetAttribLocation(program, "inTex");
    glEnableVertexAttribArray(posTex);
    glVertexAttribPointer(posTex, 2, GL_FLOAT, GL_FALSE, 0, texCoords);

    // Draw with level 0, the whole framebuffer should be Red.
    glViewport(0, 0, getWindowWidth(), getWindowHeight());
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_GL_NO_ERROR();
    EXPECT_PIXEL_COLOR_EQ(0, 0, GLColor::red);
    EXPECT_PIXEL_COLOR_EQ(1, 1, GLColor::red);
    EXPECT_PIXEL_COLOR_EQ(getWindowWidth() - 1, getWindowHeight() - 1, GLColor::red);
    // Draw with level 1, a quarter of the framebuffer should be Green.
    glViewport(0, 0, getWindowWidth() / 2, getWindowHeight() / 2);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_PIXEL_COLOR_EQ(0, 0, GLColor::green);
    EXPECT_PIXEL_COLOR_EQ(1, 1, GLColor::green);
    EXPECT_PIXEL_COLOR_EQ(getWindowWidth() / 2 - 1, getWindowHeight() / 2 - 1, GLColor::green);

    // Clear the texture level 0's (0, 0) position to Blue.
    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
    ANGLE_GL_COMPUTE_PROGRAM(csProgram, kCSSource);
    glUseProgram(csProgram);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
    EXPECT_GL_NO_ERROR();
    glFinish();

    glUseProgram(program);
    // Draw with level 0, the first position should be Blue.
    glViewport(0, 0, getWindowWidth(), getWindowHeight());
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_PIXEL_COLOR_EQ(0, 0, GLColor::blue);
    EXPECT_PIXEL_COLOR_EQ(1, 1, GLColor::red);
    EXPECT_PIXEL_COLOR_EQ(getWindowWidth() - 1, getWindowHeight() - 1, GLColor::red);
    // Draw with level 1, a quarter of the framebuffer should be Green.
    glViewport(0, 0, getWindowWidth() / 2, getWindowHeight() / 2);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_PIXEL_COLOR_EQ(0, 0, GLColor::green);
    EXPECT_PIXEL_COLOR_EQ(1, 1, GLColor::green);
    EXPECT_PIXEL_COLOR_EQ(getWindowWidth() / 2 - 1, getWindowHeight() / 2 - 1, GLColor::green);
}

// Test that writes via compute are visible to draw call
TEST_P(ComputeShaderTest, DispatchDrawBlend)
{
    const char kCS[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1) in;
layout(rgba8, binding = 0) writeonly uniform highp image2D image;
uniform vec4 color;
void main()
{
    imageStore(image, ivec2(0, 0), color);
})";

    ANGLE_GL_COMPUTE_PROGRAM(imageWrite, kCS);
    ANGLE_GL_PROGRAM(drawRed, essl1_shaders::vs::Simple(), essl1_shaders::fs::Red());

    glUseProgram(imageWrite);
    const GLint uniLoc = glGetUniformLocation(imageWrite, "color");

    for (const GLboolean layered : {GL_FALSE, GL_TRUE})
    {
        GLTexture texture;
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 1, 1);

        GLFramebuffer fbo;
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);

        // Leave the texture intentionally uninitialized, it's initialized by the compute shader.
        glBindImageTexture(0, texture, 0, layered, 0, GL_WRITE_ONLY, GL_RGBA8);

        // Use with compute shader once; in the Vulkan backend, this may recreate the image's
        // backing storage (which at the time of writing this comment, copies the undefined values
        // from the existing image into a new image, marking it as having defined values).
        // glInvalidateFramebuffer is used to mark the contents undefined again.
        glUseProgram(imageWrite);
        glUniform4f(uniLoc, 0, 1, 0, 1);
        glDispatchCompute(1, 1, 1);
        EXPECT_GL_NO_ERROR();

        constexpr std::array<GLenum, 1> kDiscard = {GL_COLOR_ATTACHMENT0};
        glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
        glInvalidateFramebuffer(GL_FRAMEBUFFER, 1, kDiscard.data());

        // Dispatch again, which is supposed to make the contents defined.
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        glUniform4f(uniLoc, 0, 0, 1, 1);
        glDispatchCompute(1, 1, 1);
        EXPECT_GL_NO_ERROR();

        // Blend into it after the CS write
        glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);

        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE);
        drawQuad(drawRed, essl1_shaders::PositionAttrib(), 0.0f);

        EXPECT_PIXEL_COLOR_EQ(0, 0, GLColor::magenta) << "Layered? " << layered;
        ASSERT_GL_NO_ERROR();
    }
}

// Test that maxComputeWorkGroupCount is valid number.
TEST_P(ComputeShaderTest, ValidateMaxComputeWorkGroupCount)
{
    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=1) in;
void main()
{
})";

    GLuint program = glCreateProgram();
    GLuint cs      = CompileShader(GL_COMPUTE_SHADER, kCS);
    EXPECT_NE(0u, cs);

    glAttachShader(program, cs);
    glDeleteShader(cs);

    GLint x, y, z;
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &x);
    EXPECT_LE(65535, x);
    EXPECT_GE(std::numeric_limits<GLint>::max(), x);

    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &y);
    EXPECT_LE(65535, y);
    EXPECT_GE(std::numeric_limits<GLint>::max(), y);

    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &z);
    EXPECT_LE(65535, z);
    EXPECT_GE(std::numeric_limits<GLint>::max(), z);

    glDeleteProgram(program);
    EXPECT_GL_NO_ERROR();
}

// Validate that on Vulkan, compute pipeline driver uniforms descriptor set is updated after an
// internal compute-based UtilsVk function is used.  The latter is achieved through a draw with a
// vertex buffer whose format is not natively supported.  Atomic counters are used to make sure the
// compute pipeline uses the driver uniforms descriptor set.
TEST_P(ComputeShaderTest, DispatchConvertVertexDispatch)
{
    ANGLE_SKIP_TEST_IF(!IsGLExtensionEnabled("GL_OES_vertex_type_10_10_10_2"));

    constexpr uint32_t kVertexCount = 6;

    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=6, local_size_y=1, local_size_z=1) in;

layout(binding = 0) uniform atomic_uint ac;

layout(binding=0, std140) buffer VertexData
{
    uint data[];
};

void main()
{
    atomicCounterIncrement(ac);
    data[gl_GlobalInvocationID.x] = gl_GlobalInvocationID.x;
})";

    constexpr char kVS[] = R"(#version 310 es
precision mediump float;

layout(location = 0) in vec4 position;
layout(location = 1) in uvec4 data;

out vec4 color;

void main() {
    color = data.x < 6u && data.y == 0u && data.z == 0u && data.w == 0u
        ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
    gl_Position = position;
})";

    constexpr char kFS[] = R"(#version 310 es
precision mediump float;
in vec4 color;
out vec4 colorOut;
void main() {
    colorOut = color;
})";

    ANGLE_GL_COMPUTE_PROGRAM(programCS, kCS);
    ANGLE_GL_PROGRAM(programVSFS, kVS, kFS);
    EXPECT_GL_NO_ERROR();

    // Create atomic counter buffer
    GLBuffer atomicCounterBuffer;
    constexpr GLuint kInitialAcbData = 0;
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, atomicCounterBuffer);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(kInitialAcbData), &kInitialAcbData,
                 GL_STATIC_DRAW);
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, atomicCounterBuffer);
    EXPECT_GL_NO_ERROR();

    // Create vertex buffer
    constexpr unsigned kVertexBufferInitData[kVertexCount] = {};
    GLBuffer vertexBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(kVertexBufferInitData), kVertexBufferInitData,
                 GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vertexBuffer);
    EXPECT_GL_NO_ERROR();

    // Create position buffer
    constexpr GLfloat positions[kVertexCount * 2] = {1.0, 1.0, -1.0, 1.0,  -1.0, -1.0,
                                                     1.0, 1.0, -1.0, -1.0, 1.0,  -1.0};
    GLBuffer positionBuffer;
    glBindBuffer(GL_ARRAY_BUFFER, positionBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);
    EXPECT_GL_NO_ERROR();

    // Create vertex array
    GLVertexArray vao;
    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, positionBuffer);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, false, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_UNSIGNED_INT_10_10_10_2_OES, false, 0, 0);
    EXPECT_GL_NO_ERROR();

    // Fill the vertex buffer with a dispatch call
    glUseProgram(programCS);
    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);

    // Draw using the vertex buffer, causing vertex format conversion in compute (in the Vulkan
    // backend)
    glUseProgram(programVSFS);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, kVertexCount);
    EXPECT_GL_NO_ERROR();

    // Issue another dispatch call. The driver uniforms descriptor set must be rebound.
    glUseProgram(programCS);
    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    // Verify that the atomic counter has the expected value.
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, atomicCounterBuffer);
    auto mappedBuffer =
        MapBufferRange<const GLuint>(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), GL_MAP_READ_BIT);
    ASSERT_FALSE(mappedBuffer.empty());
    EXPECT_EQ(kVertexCount * 2, mappedBuffer[0]);
    glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
}

// This test mirrors a dEQP test that is failing on some platforms
//  - KHR-GLES31.core.shader_image_load_store.advanced-sync-vertexArray
// The test has three pipelines:
//  - First uses a vertex shader to store data to images
//  - Second uses a compute shader to read from those images and store to buffers
//  - Third uses those buffers as vertex input to render a color
// It does this twice with different colors, and we were failing the second pass.
// http://anglebug.com/416573908
TEST_P(ComputeShaderTest, AdvancedSyncVertexArray)
{
    GLint maxVertexImageUniforms = 0;
    glGetIntegerv(GL_MAX_VERTEX_IMAGE_UNIFORMS, &maxVertexImageUniforms);
    ANGLE_SKIP_TEST_IF(maxVertexImageUniforms < 3);

    constexpr char kCopyCS[] = R"(#version 310 es
#define KSIZE 4
layout (local_size_x = KSIZE) in;

layout(rgba32f, binding = 0) readonly uniform highp image2D  positionImage;
layout(rgba32f, binding = 1) readonly uniform highp image2D  colorImage;
layout(r32ui,   binding = 2) readonly uniform highp uimage2D elementImage;

layout(std430, binding = 1) buffer positionBuffer {
    vec2 position[KSIZE];
};
layout(std430, binding = 2) buffer colorBuffer {
    vec4 color[KSIZE];
};
layout(std430, binding = 3) buffer elementBuffer {
    uint element[KSIZE];
};

void main() {
    ivec2 coord = ivec2(gl_LocalInvocationID.x, 0);

    position[coord.x] = imageLoad(positionImage, coord).xy;
       color[coord.x] = imageLoad(colorImage, coord);
     element[coord.x] = uint(imageLoad(elementImage, coord).x);
})";

    constexpr char kStoreVS[] = R"(#version 310 es
layout(rgba32f, binding = 0) writeonly uniform highp image2D positionImage;
layout(rgba32f, binding = 1) writeonly uniform highp image2D colorImage;
layout(r32ui,   binding = 2) writeonly uniform highp uimage2D elementImage;

uniform vec4 inColor;

void main() {
    vec2[4] data = vec2[4](vec2(-1, -1), vec2(1, -1), vec2(-1, 1), vec2(1, 1));

    imageStore(positionImage, ivec2(gl_VertexID, 0), vec4(data[gl_VertexID], 0.0, 1.0));
    imageStore(colorImage,    ivec2(gl_VertexID, 0), inColor);
    imageStore(elementImage,  ivec2(gl_VertexID, 0), uvec4(gl_VertexID));
})";

    constexpr char kStoreFS[] = R"(#version 310 es
precision mediump float;

void main() {
    discard;
})";

    constexpr char kDrawVS[] = R"(#version 310 es
layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec4 inColor;

out vec4 vsColor;

void main() {
    gl_Position = inPosition;
    vsColor     = inColor;
})";

    constexpr char kDrawFS[] = R"(#version 310 es
precision mediump float;

in vec4 vsColor;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vsColor;
})";

    // The store program writes values to images
    ANGLE_GL_PROGRAM(storeProgram, kStoreVS, kStoreFS);
    // The compute program copies from images to buffers
    ANGLE_GL_COMPUTE_PROGRAM(copyProgram, kCopyCS);
    // The draw program reads from vertex input buffers
    ANGLE_GL_PROGRAM(drawProgram, kDrawVS, kDrawFS);
    EXPECT_GL_NO_ERROR();

    GLint inColorLoc = glGetUniformLocation(storeProgram, "inColor");
    ASSERT_NE(inColorLoc, -1);

    GLuint texture[3];
    glGenTextures(3, texture);
    glBindTexture(GL_TEXTURE_2D, texture[0]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, 4, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, texture[1]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, 4, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, texture[2]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, 4, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    EXPECT_GL_NO_ERROR();

    GLuint attribless_vao;
    GLuint draw_vao;
    glGenVertexArrays(1, &attribless_vao);
    glGenVertexArrays(1, &draw_vao);
    glBindVertexArray(draw_vao);
    EXPECT_GL_NO_ERROR();

    GLuint position_buffer;
    glGenBuffers(1, &position_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, position_buffer);
    glBufferData(GL_ARRAY_BUFFER, 4 * 4 * 4, 0, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    EXPECT_GL_NO_ERROR();

    GLuint color_buffer;
    glGenBuffers(1, &color_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, color_buffer);
    glBufferData(GL_ARRAY_BUFFER, 4 * 4 * 4, 0, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    EXPECT_GL_NO_ERROR();

    GLuint element_buffer;
    glGenBuffers(1, &element_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 * 4 * 4, 0, GL_STATIC_DRAW);
    EXPECT_GL_NO_ERROR();

    glBindVertexArray(0);
    // Vertex position
    glBindImageTexture(0, texture[0], 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
    // Vertex color
    glBindImageTexture(1, texture[1], 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
    // Vertex ID
    glBindImageTexture(2, texture[2], 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
    EXPECT_GL_NO_ERROR();

    // Store Vertex Positions, GREEN, and Vertex ID to the images
    glUseProgram(storeProgram);
    glUniform4f(inColorLoc, 0.0f, 1.0f, 0.0f, 1.0f);
    glBindVertexArray(attribless_vao);
    glDrawArrays(GL_POINTS, 0, 4);
    EXPECT_GL_NO_ERROR();

    // Copy GREEN from images to vertex array buffers
    glUseProgram(copyProgram);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, position_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, color_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, element_buffer);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    // Draw GREEN using the incoming vertex data
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(drawProgram);
    glBindVertexArray(draw_vao);
    glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);
    glDrawElements(GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_INT, 0);
    EXPECT_GL_NO_ERROR();

    // Check for GREEN
    EXPECT_PIXEL_COLOR_EQ(0, 0, GLColor::green);

    // Store Vertex Positions, BLUE, and Vertex ID to the images
    glUseProgram(storeProgram);
    glUniform4f(inColorLoc, 0.0f, 0.0f, 1.0f, 1.0f);
    glBindVertexArray(attribless_vao);
    glDrawArrays(GL_POINTS, 0, 4);
    EXPECT_GL_NO_ERROR();

    // Copy BLUE from images to vertex array buffers
    glUseProgram(copyProgram);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    // Draw BLUE using incoming vertex data
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(drawProgram);
    glBindVertexArray(draw_vao);
    glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);
    glDrawElements(GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_INT, 0);
    EXPECT_GL_NO_ERROR();

    // Check for BLUE
    EXPECT_PIXEL_COLOR_EQ(0, 0, GLColor::blue);
}

// Validate that on Vulkan, compute pipeline is correctly bound after an internal dispatch call is
// made.  Blit stencil may issue a dispatch call.
TEST_P(ComputeShaderTest, DispatchBlitStencilDispatch)
{
    // http://anglebug.com/42264069
    ANGLE_SKIP_TEST_IF(IsQualcomm() && IsOpenGLES());

    // http://anglebug.com/42263641
    ANGLE_SKIP_TEST_IF(IsIntel() && IsLinux() && IsOpenGL());

    constexpr GLsizei kSize = 1;

    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=6, local_size_y=1, local_size_z=1) in;

uniform vec4 data;

layout(rgba8, binding = 0) writeonly uniform highp image2D image;

void main()
{
    imageStore(image, ivec2(gl_LocalInvocationID.xy), data);
})";

    ANGLE_GL_COMPUTE_PROGRAM(programCS, kCS);
    EXPECT_GL_NO_ERROR();

    // Create a framebuffer with stencil buffer.  Use multisampled textures so the future blit
    // cannot use vkCmdBlitImage.
    GLTexture color;
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, color);
    glTexStorage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA8, kSize, kSize, true);

    GLTexture depthStencil;
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, depthStencil);
    glTexStorage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_DEPTH24_STENCIL8, kSize, kSize,
                              true);

    GLFramebuffer fbo;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, color,
                           0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D_MULTISAMPLE,
                           depthStencil, 0);
    ASSERT_GL_FRAMEBUFFER_COMPLETE(GL_FRAMEBUFFER);
    ASSERT_GL_NO_ERROR();

    // Clear the stencil and make sure it's done.
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClearStencil(0x55);
    glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    glEnable(GL_STENCIL_TEST);
    glStencilFunc(GL_EQUAL, 0x55, 0xFF);
    glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
    glStencilMask(0xFF);

    ANGLE_GL_PROGRAM(drawRed, essl1_shaders::vs::Simple(), essl1_shaders::fs::Red());
    drawQuad(drawRed, essl1_shaders::PositionAttrib(), 0.0f);
    ASSERT_GL_NO_ERROR();

    GLTexture colorCopy;
    glBindTexture(GL_TEXTURE_2D, colorCopy);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, kSize, kSize);

    GLFramebuffer copyFbo;
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, copyFbo);
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorCopy, 0);
    ASSERT_GL_FRAMEBUFFER_COMPLETE(GL_DRAW_FRAMEBUFFER);
    glBlitFramebuffer(0, 0, kSize, kSize, 0, 0, kSize, kSize, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    ASSERT_GL_NO_ERROR();

    glBindFramebuffer(GL_READ_FRAMEBUFFER, copyFbo);
    EXPECT_PIXEL_COLOR_EQ(0, 0, GLColor::red);

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    // Setup image for compute call
    GLTexture computeOut;
    glBindTexture(GL_TEXTURE_2D, computeOut);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, kSize, kSize);
    glBindImageTexture(0, computeOut, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
    ASSERT_GL_NO_ERROR();

    // Issue a dispatch call.
    glUseProgram(programCS);
    GLint uniformLoc = glGetUniformLocation(programCS, "data");
    ASSERT_NE(uniformLoc, -1);

    glUniform4f(uniformLoc, 0.0f, 0.0f, 1.0f, 1.0f);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    ASSERT_GL_NO_ERROR();

    // Blit the stencil texture.  This may use a compute shader internally.
    GLTexture depthStencilCopy;
    glBindTexture(GL_TEXTURE_2D, depthStencilCopy);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH24_STENCIL8, kSize, kSize);
    ASSERT_GL_NO_ERROR();

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, copyFbo);
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D,
                           depthStencilCopy, 0);
    ASSERT_GL_FRAMEBUFFER_COMPLETE(GL_DRAW_FRAMEBUFFER);
    ASSERT_GL_NO_ERROR();

    glBlitFramebuffer(0, 0, kSize, kSize, 0, 0, kSize, kSize, GL_STENCIL_BUFFER_BIT, GL_NEAREST);
    ASSERT_GL_NO_ERROR();

    // Issue another dispatch call.
    glUniform4f(uniformLoc, 0.0f, 1.0f, 0.0f, 1.0f);
    glDispatchCompute(1, 1, 1);
    ASSERT_GL_NO_ERROR();

    // Verify the results.
    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, copyFbo);
    glBindTexture(GL_TEXTURE_2D, computeOut);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, computeOut, 0);
    ASSERT_GL_NO_ERROR();

    EXPECT_PIXEL_COLOR_EQ(0, 0, GLColor::green);

    // Verify the blit copy results.
    drawQuad(drawRed, essl1_shaders::PositionAttrib(), 0.0f);
    ASSERT_GL_NO_ERROR();

    EXPECT_PIXEL_COLOR_EQ(0, 0, GLColor::red);
}

// Validate that on Vulkan, compute pipeline is correctly bound after an internal dispatch call is
// made.  Generate mipmap may issue a dispatch call.
TEST_P(ComputeShaderTest, DispatchGenerateMipmapDispatch)
{
    constexpr GLsizei kSize = 8;

    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=6, local_size_y=1, local_size_z=1) in;

uniform vec4 data;

layout(rgba8, binding = 0) writeonly uniform highp image2D image;

void main()
{
    imageStore(image, ivec2(gl_LocalInvocationID.xy), data);
})";

    ANGLE_GL_COMPUTE_PROGRAM(programCS, kCS);
    EXPECT_GL_NO_ERROR();

    GLTexture color;
    glBindTexture(GL_TEXTURE_2D, color);
    glTexStorage2D(GL_TEXTURE_2D, 4, GL_RGBA8, kSize, kSize);

    const std::vector<GLColor> kInitialColor(kSize * kSize, GLColor::green);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kSize, kSize, GL_RGBA, GL_UNSIGNED_BYTE,
                    kInitialColor.data());

    // Setup image for compute call
    GLTexture computeOut;
    glBindTexture(GL_TEXTURE_2D, computeOut);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, kSize, kSize);
    glBindImageTexture(0, computeOut, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
    ASSERT_GL_NO_ERROR();

    // Issue a dispatch call.
    glUseProgram(programCS);
    GLint uniformLoc = glGetUniformLocation(programCS, "data");
    ASSERT_NE(uniformLoc, -1);

    glUniform4f(uniformLoc, 0.0f, 0.0f, 1.0f, 1.0f);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    ASSERT_GL_NO_ERROR();

    // Generate mipmap on the texture.  This may use a compute shader internally.
    glBindTexture(GL_TEXTURE_2D, color);
    glGenerateMipmap(GL_TEXTURE_2D);

    // Issue another dispatch call.
    glUniform4f(uniformLoc, 0.0f, 1.0f, 0.0f, 1.0f);
    glDispatchCompute(1, 1, 1);
    ASSERT_GL_NO_ERROR();

    // Verify the results.
    GLFramebuffer fbo;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glBindTexture(GL_TEXTURE_2D, computeOut);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, computeOut, 0);
    ASSERT_GL_NO_ERROR();

    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
    EXPECT_PIXEL_COLOR_EQ(0, 0, GLColor::green);
}

// Write to image array with an aliasing format.
TEST_P(ComputeShaderTest, AliasingFormatForImageArray)
{
    // http://anglebug.com/42263894
    ANGLE_SKIP_TEST_IF(IsD3D11());

    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=2) in;
layout(r32ui, binding = 0) writeonly uniform highp uimage2DArray image;
void main()
{
    uint yellow = 0xFF00FFFFu;
    imageStore(image, ivec3(gl_LocalInvocationID.xyz), uvec4(yellow, 0, 0, 0));
})";

    constexpr int kWidth = 1, kHeight = 1, kDepth = 2;

    const std::vector<GLColor> kInitData(kWidth * kHeight * kDepth, GLColor::black);

    GLTexture texture;
    glBindTexture(GL_TEXTURE_2D_ARRAY, texture);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGBA8, kWidth, kHeight, kDepth);
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, kWidth, kHeight, kDepth, GL_RGBA,
                    GL_UNSIGNED_BYTE, kInitData.data());
    EXPECT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    glUseProgram(program);

    // Output yellow to both layers.
    glBindImageTexture(0, texture, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32UI);
    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    // Verify results.
    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
    EXPECT_GL_NO_ERROR();

    GLFramebuffer framebuffer;
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
    glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture, 0, 0);
    glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, texture, 0, 1);
    EXPECT_GL_NO_ERROR();

    glReadBuffer(GL_COLOR_ATTACHMENT0);
    EXPECT_PIXEL_COLOR_EQ(0, 0, GLColor::yellow);

    glReadBuffer(GL_COLOR_ATTACHMENT1);
    EXPECT_PIXEL_COLOR_EQ(0, 0, GLColor::yellow);
}

// Write to one layer of image array with an aliasing format.
TEST_P(ComputeShaderTest, AliasingFormatForOneLayerOfImageArray)
{
    // http://anglebug.com/42263894
    ANGLE_SKIP_TEST_IF(IsD3D11());

    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(r32ui, binding = 0) writeonly uniform highp uimage2D image;
void main()
{
    uint yellow = 0xFF00FFFFu;
    imageStore(image, ivec2(gl_LocalInvocationID.xy), uvec4(yellow, 0, 0, 0));
})";

    constexpr int kWidth = 1, kHeight = 1, kDepth = 2;

    const std::vector<GLColor> kInitData(kWidth * kHeight * kDepth, GLColor::black);

    GLTexture texture;
    glBindTexture(GL_TEXTURE_2D_ARRAY, texture);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGBA8, kWidth, kHeight, kDepth);
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, kWidth, kHeight, kDepth, GL_RGBA,
                    GL_UNSIGNED_BYTE, kInitData.data());
    EXPECT_GL_NO_ERROR();

    GLFramebuffer framebuffer;
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
    glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture, 0, 0);
    glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, texture, 0, 1);
    EXPECT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    glUseProgram(program);

    // Output yellow to layer 0.
    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    // Verify that only layer 0 was changed.
    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
    EXPECT_GL_NO_ERROR();

    glReadBuffer(GL_COLOR_ATTACHMENT0);
    EXPECT_PIXEL_COLOR_EQ(0, 0, GLColor::yellow);

    glReadBuffer(GL_COLOR_ATTACHMENT1);
    EXPECT_PIXEL_COLOR_EQ(0, 0, GLColor::black);

    // Reset texture back to black.
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, kWidth, kHeight, kDepth, GL_RGBA,
                    GL_UNSIGNED_BYTE, kInitData.data());

    // Output yellow to layer 1.
    glBindImageTexture(0, texture, 0, GL_FALSE, 1, GL_WRITE_ONLY, GL_R32UI);
    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    // Verify that only layer 1 was changed.
    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
    EXPECT_GL_NO_ERROR();

    glReadBuffer(GL_COLOR_ATTACHMENT0);
    EXPECT_PIXEL_COLOR_EQ(0, 0, GLColor::black);

    glReadBuffer(GL_COLOR_ATTACHMENT1);
    EXPECT_PIXEL_COLOR_EQ(0, 0, GLColor::yellow);
}

// Test glMemoryBarrier(CLIENT_MAPPED_BUFFER_BARRIER_BIT_EXT) by writing to persistenly mapped
// buffer from a compute shader.
TEST_P(ComputeShaderTest, WriteToPersistentBuffer)
{
    ANGLE_SKIP_TEST_IF(!IsGLExtensionEnabled("GL_EXT_buffer_storage"));

    constexpr char kCS[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(std140, binding = 0) buffer block {
    uvec4 data;
} outBlock;
void main()
{
    outBlock.data += uvec4(1);
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kCS);
    glUseProgram(program);

    constexpr std::array<uint32_t, 4> kInitData = {};

    GLBuffer coherentBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, coherentBuffer);
    glBufferStorageEXT(
        GL_SHADER_STORAGE_BUFFER, sizeof(kInitData), kInitData.data(),
        GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT_EXT | GL_MAP_COHERENT_BIT_EXT);

    GLBuffer nonCoherentBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, nonCoherentBuffer);
    glBufferStorageEXT(GL_SHADER_STORAGE_BUFFER, sizeof(kInitData), kInitData.data(),
                       GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT_EXT);

    // Map the buffers for read and write.
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, coherentBuffer);
    auto coherentMapped = MapBufferRange<uint32_t>(
        GL_SHADER_STORAGE_BUFFER, 0, sizeof(kInitData),
        GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT_EXT | GL_MAP_COHERENT_BIT_EXT);
    ASSERT_GL_NO_ERROR();
    ASSERT_FALSE(coherentMapped.empty());

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, nonCoherentBuffer);
    auto nonCoherentMapped =
        MapBufferRange<uint32_t>(GL_SHADER_STORAGE_BUFFER, 0, sizeof(kInitData),
                                 GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT_EXT);
    ASSERT_GL_NO_ERROR();
    ASSERT_FALSE(nonCoherentMapped.empty());

    constexpr std::array<uint32_t, 4> kCoherentExpectedData = {
        0x12354678u,
        0x2468ACE0u,
        0x13579BDFu,
        0x76543210u,
    };

    constexpr std::array<uint32_t, 4> kNonCoherentExpectedData = {
        0x9ABCDEF0u,
        0xFDB97531u,
        0x1F2E3D4Bu,
        0x5A697887u,
    };

    coherentMapped[0] = kCoherentExpectedData[0] - 1;
    coherentMapped[1] = kCoherentExpectedData[1] - 1;
    coherentMapped[2] = kCoherentExpectedData[2] - 1;
    coherentMapped[3] = kCoherentExpectedData[3] - 1;

    nonCoherentMapped[0] = kNonCoherentExpectedData[0] - 1;
    nonCoherentMapped[1] = kNonCoherentExpectedData[1] - 1;
    nonCoherentMapped[2] = kNonCoherentExpectedData[2] - 1;
    nonCoherentMapped[3] = kNonCoherentExpectedData[3] - 1;

    // Test coherent write
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, coherentBuffer);
    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();

    glFinish();
    EXPECT_EQ(coherentMapped[0], kCoherentExpectedData[0]);
    EXPECT_EQ(coherentMapped[1], kCoherentExpectedData[1]);
    EXPECT_EQ(coherentMapped[2], kCoherentExpectedData[2]);
    EXPECT_EQ(coherentMapped[3], kCoherentExpectedData[3]);

    // Test non-coherent write
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, nonCoherentBuffer);
    glDispatchCompute(1, 1, 1);

    glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT_EXT);
    EXPECT_GL_NO_ERROR();

    glFinish();
    EXPECT_EQ(nonCoherentMapped[0], kNonCoherentExpectedData[0]);
    EXPECT_EQ(nonCoherentMapped[1], kNonCoherentExpectedData[1]);
    EXPECT_EQ(nonCoherentMapped[2], kNonCoherentExpectedData[2]);
    EXPECT_EQ(nonCoherentMapped[3], kNonCoherentExpectedData[3]);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, coherentBuffer);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, nonCoherentBuffer);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    EXPECT_GL_NO_ERROR();
}

// Verify the CS doesn't overwrite the mapped buffer data.
TEST_P(ComputeShaderTest, ImageBufferMapWrite)
{
    ANGLE_SKIP_TEST_IF(!IsGLExtensionEnabled("GL_OES_texture_buffer"));

    constexpr char kComputeImageBuffer[] = R"(#version 310 es
#extension GL_OES_texture_buffer : require
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(rgba32f, binding = 0) uniform highp writeonly imageBuffer dst;
uniform vec4 uniformData;
void main()
{
    imageStore(dst, int(gl_GlobalInvocationID.x), uniformData);
})";

    GLProgram program;
    program.makeCompute(kComputeImageBuffer);
    glUseProgram(program);

    GLBuffer textureBufferStorage;
    GLTexture texture;
    constexpr std::array<float, 4> kInitData = {1.0, 0.0, 0.0, 1.0};

    glBindBuffer(GL_TEXTURE_BUFFER, textureBufferStorage);
    glBufferData(GL_TEXTURE_BUFFER, sizeof(kInitData), kInitData.data(), GL_STATIC_DRAW);
    EXPECT_GL_NO_ERROR();

    glBindTexture(GL_TEXTURE_BUFFER, texture);
    glTexBufferEXT(GL_TEXTURE_BUFFER, GL_RGBA32F, textureBufferStorage);
    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
    EXPECT_GL_NO_ERROR();

    constexpr std::array<float, 4> kComputeShaderData = {0.0, 1.0, 0.0, 1.0};
    GLint uniformLocation = glGetUniformLocation(program, "uniformData");
    ASSERT_NE(uniformLocation, -1);
    glUniform4f(uniformLocation, kComputeShaderData[0], kComputeShaderData[1],
                kComputeShaderData[2], kComputeShaderData[3]);
    EXPECT_GL_NO_ERROR();

    // Write to the buffer with the CS.
    glDispatchCompute(1, 1, 1);

    // Issue the appropriate memory barrier before mapping the buffer.
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    // Map the buffer and write to it.
    constexpr std::array<float, 4> kMapData = {0.0, 0.0, 1.0, 1.0};
    auto writeBuffer =
        MapBufferRange<float>(GL_TEXTURE_BUFFER, 0, sizeof(kMapData), GL_MAP_WRITE_BIT);
    ASSERT_FALSE(writeBuffer.empty());
    writeBuffer.copy_from(Span(kMapData));
    glUnmapBuffer(GL_TEXTURE_BUFFER);

    glFinish();

    // Read back and verify buffer data.
    auto mappedBuffer =
        MapBufferRange<const float>(GL_TEXTURE_BUFFER, 0, sizeof(kMapData), GL_MAP_READ_BIT);
    EXPECT_EQ(Span(kMapData), mappedBuffer);
    glUnmapBuffer(GL_TEXTURE_BUFFER);
}

// Test compute shader write to texture buffer followed by texSubData and followed by compute shader
// write to texture buffer again.
TEST_P(ComputeShaderTest, ImageBufferMapWriteAndBufferSubData)
{
    ANGLE_SKIP_TEST_IF(!IsGLExtensionEnabled("GL_OES_texture_buffer"));

    constexpr char kComputeImageBuffer[] = R"(#version 310 es
#extension GL_OES_texture_buffer : require
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(rgba32f, binding = 0) uniform highp writeonly imageBuffer dst;
uniform vec4 uniformData;
uniform int uniformOffset;
void main()
{
    imageStore(dst, uniformOffset, uniformData);
})";

    GLProgram program;
    program.makeCompute(kComputeImageBuffer);
    glUseProgram(program);

    for (int loop = 0; loop < 2; loop++)
    {
        GLBuffer textureBufferStorage;
        GLTexture texture;
        constexpr unsigned int kShaderUsedSize    = sizeof(float) * 4;
        constexpr unsigned int kMiddlePaddingSize = 1024;
        constexpr unsigned int kBufferSize = kShaderUsedSize + kMiddlePaddingSize + kShaderUsedSize;
        constexpr unsigned int kOffset0    = 0;
        constexpr unsigned int kOffset1    = kShaderUsedSize;
        constexpr unsigned int kOffset2    = kShaderUsedSize + kMiddlePaddingSize;

        glBindBuffer(GL_TEXTURE_BUFFER, textureBufferStorage);
        glBufferData(GL_TEXTURE_BUFFER, kBufferSize, nullptr, GL_STATIC_DRAW);

        glBindTexture(GL_TEXTURE_BUFFER, texture);
        glTexBufferEXT(GL_TEXTURE_BUFFER, GL_RGBA32F, textureBufferStorage);
        glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);

        // Write to the buffer with the CS.
        constexpr std::array<float, 4> kComputeShaderData1 = {0.0, 1.0, 0.0, 1.0};
        GLint uniformDataLocation = glGetUniformLocation(program, "uniformData");
        ASSERT_NE(uniformDataLocation, -1);
        glUniform4f(uniformDataLocation, kComputeShaderData1[0], kComputeShaderData1[1],
                    kComputeShaderData1[2], kComputeShaderData1[3]);
        GLint uniformOffsetLocation = glGetUniformLocation(program, "uniformOffset");
        ASSERT_NE(uniformOffsetLocation, -1);
        glUniform1i(uniformOffsetLocation, kOffset0 / (sizeof(float) * 4));
        glDispatchCompute(1, 1, 1);
        // Issue the appropriate memory barrier before mapping the buffer.
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        // Write to the buffer with the CS.
        constexpr std::array<float, 4> kComputeShaderData2 = {1.0, 0.0, 1.0, 1.0};
        glUniform4f(uniformDataLocation, kComputeShaderData2[0], kComputeShaderData2[1],
                    kComputeShaderData2[2], kComputeShaderData2[3]);
        glUniform1i(uniformOffsetLocation, kOffset2 / (sizeof(float) * 4));
        glDispatchCompute(1, 1, 1);

        if (loop == 1)
        {
            // Make write operation finished but read operation pending. We don't care about the
            // actual rendering result but just to have unflushed commands using the buffer so that
            // it appears as pending.
            glFinish();
            constexpr char kVS[] = R"(attribute vec4 in_attrib;
                                    varying vec4 v_attrib;
                                    void main()
                                    {
                                        v_attrib = in_attrib;
                                        gl_Position = vec4(0.0, 0.0, 0.5, 1.0);
                                        gl_PointSize = 100.0;
                                    })";
            constexpr char kFS[] = R"(precision mediump float;
                                    varying vec4 v_attrib;
                                    void main()
                                    {
                                        gl_FragColor = v_attrib;
                                    })";
            GLuint readProgram   = CompileProgram(kVS, kFS);
            ASSERT_NE(readProgram, 0U);
            GLint attribLocation = glGetAttribLocation(readProgram, "in_attrib");
            ASSERT_NE(attribLocation, -1);
            glUseProgram(readProgram);
            ASSERT_GL_NO_ERROR();
            glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);
            glBindBuffer(GL_ARRAY_BUFFER, textureBufferStorage);
            glVertexAttribPointer(attribLocation, 4, GL_UNSIGNED_BYTE, GL_TRUE, 4, nullptr);
            glEnableVertexAttribArray(attribLocation);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, textureBufferStorage);
            glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
            glDrawElements(GL_POINTS, 1, GL_UNSIGNED_INT, nullptr);
            ASSERT_GL_NO_ERROR();
        }

        // Use subData to update middle portion of data to trigger acquireAndUpdate code path in
        // ANGLE
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        glBindBuffer(GL_TEXTURE_BUFFER, textureBufferStorage);
        constexpr unsigned int kMiddlePaddingValue = 0x55555555u;
        std::vector<unsigned int> kPaddingValues(kMiddlePaddingSize / sizeof(unsigned int),
                                                 kMiddlePaddingValue);
        glBufferSubData(GL_TEXTURE_BUFFER, kOffset1, kMiddlePaddingSize, kPaddingValues.data());

        // Read back and verify buffer data.
        ANGLE_ENABLE_STRUCT_PADDING_WARNINGS
        struct MappedData
        {
            std::array<GLfloat, 4> shaderData1;
            std::array<GLuint, kMiddlePaddingSize / sizeof(GLuint)> padding;
            std::array<GLfloat, 4> shaderData2;
        };
        ANGLE_DISABLE_STRUCT_PADDING_WARNINGS
        auto mappedBuffer =
            MapBufferRange<const MappedData>(GL_TEXTURE_BUFFER, 0, kBufferSize, GL_MAP_READ_BIT);
        ASSERT_FALSE(mappedBuffer.empty());
        const MappedData &mappedData = mappedBuffer[0];

        EXPECT_EQ(mappedData.shaderData1, kComputeShaderData1);
        for (GLuint value : mappedData.padding)
        {
            EXPECT_EQ(value, kMiddlePaddingValue);
        }
        EXPECT_EQ(mappedData.shaderData2, kComputeShaderData2);

        glUnmapBuffer(GL_TEXTURE_BUFFER);
        EXPECT_GL_NO_ERROR();
    }
}

// Test one texture sampled by fragment shader, then bind it to image, followed by compute
// shader load this image, and fragment shader read it again.
TEST_P(ComputeShaderTest, DrawDispatchImageReadDraw)
{

    constexpr char kVSSource[] = R"(#version 310 es
in vec4 a_position;
out vec2 v_texCoord;

void main()
{
    gl_Position = vec4(a_position.xy, 0.0, 1.0);
    v_texCoord = a_position.xy * 0.5 + vec2(0.5);
})";

    constexpr char kFSSource[] = R"(#version 310 es
precision mediump float;
uniform sampler2D u_tex2D;
in vec2 v_texCoord;
out vec4 out_FragColor;
void main()
{
    out_FragColor = texture(u_tex2D, v_texCoord);
})";

    constexpr char kCSSource[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(rgba32f, binding=0) readonly  uniform highp image2D uIn;
layout(std140, binding=0) buffer buf {
    vec4 outData;
};

void main()
{
    outData = imageLoad(uIn, ivec2(gl_LocalInvocationID.xy));
})";

    GLfloat initValue[4] = {1.0, 1.0, 1.0, 1.0};

    // Step 1: Set up a simple 2D Texture rendering loop.
    GLTexture texture;
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, 1, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RGBA, GL_FLOAT, initValue);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    GLBuffer vertexBuffer;
    GLfloat vertices[] = {-1, -1, 1, -1, -1, 1, 1, 1};
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(GLfloat), vertices, GL_STATIC_DRAW);

    GLBuffer ssbo;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 16, nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    ANGLE_GL_PROGRAM(program, kVSSource, kFSSource);
    glUseProgram(program);

    GLint posLoc = glGetAttribLocation(program, "a_position");
    ASSERT_NE(-1, posLoc);

    glVertexAttribPointer(posLoc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(posLoc);
    ASSERT_GL_NO_ERROR();
    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);

    glDrawArrays(GL_TRIANGLES, 0, 6);
    ASSERT_GL_NO_ERROR();

    // Step 2: load this image through compute
    ANGLE_GL_COMPUTE_PROGRAM(csProgram, kCSSource);
    glUseProgram(csProgram);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);

    glDispatchCompute(1, 1, 1);
    EXPECT_GL_NO_ERROR();
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    auto mapped = MapBufferRange<const GLfloat>(GL_SHADER_STORAGE_BUFFER, 0, 16, GL_MAP_READ_BIT);
    ASSERT_FALSE(mapped.empty());

    EXPECT_GL_NO_ERROR();
    for (GLfloat value : mapped)
    {
        EXPECT_EQ(1.0, value);
    }

    // Step3: use the first program sample texture again
    glUseProgram(program);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    ASSERT_GL_NO_ERROR();

    EXPECT_PIXEL_COLOR_EQ(1, 1, GLColor::white);
}

// Test fragment shader read a image, followed by compute shader sample it.
TEST_P(ComputeShaderTest, FSReadImageThenCSSample)
{
    GLint maxFragmentImageUniforms = 0;
    glGetIntegerv(GL_MAX_FRAGMENT_IMAGE_UNIFORMS, &maxFragmentImageUniforms);

    // MAX_FRAGMENT_IMAGE_UNIFORMS can be 0 according to OpenGL ES 3.1 SPEC.
    ANGLE_SKIP_TEST_IF(maxFragmentImageUniforms == 0);

    constexpr char kVSSource[] = R"(#version 310 es
in vec4 a_position;
out vec2 v_texCoord;
void main()
{
    gl_Position = vec4(a_position.xy, 0.0, 1.0);
    v_texCoord = a_position.xy * 0.5 + vec2(0.5);;
})";

    constexpr char kFSSource[] = R"(#version 310 es
precision mediump float;
layout(rgba32f, binding=0) readonly  uniform highp image2D uIn;
in vec2 v_texCoord;
out vec4 out_FragColor;

void main()
{
    out_FragColor = imageLoad(uIn, ivec2(v_texCoord));
})";

    constexpr char kCSSource[] = R"(#version 310 es
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(std140, binding=0) buffer buf {
    vec4 outData;
};
uniform sampler2D u_tex2D;
void main()
{
    outData = texture(u_tex2D, vec2(gl_LocalInvocationID.xy));
})";

    GLfloat initValue[4] = {1.0, 1.0, 1.0, 1.0};

    // Step 1: Set up a simple 2D Texture rendering loop.
    GLTexture texture;
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, 1, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RGBA, GL_FLOAT, initValue);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    GLBuffer vertexBuffer;
    GLfloat vertices[] = {-1, -1, 1, -1, -1, 1, 1, 1};
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(GLfloat), vertices, GL_STATIC_DRAW);

    GLBuffer ssbo;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 16, nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    ANGLE_GL_PROGRAM(program, kVSSource, kFSSource);
    glUseProgram(program);

    GLint posLoc = glGetAttribLocation(program, "a_position");
    ASSERT_NE(-1, posLoc);

    glVertexAttribPointer(posLoc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(posLoc);
    ASSERT_GL_NO_ERROR();
    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    ASSERT_GL_NO_ERROR();

    // Step 2: load this image through compute
    ANGLE_GL_COMPUTE_PROGRAM(csProgram, kCSSource);
    glUseProgram(csProgram);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);

    glDispatchCompute(1, 1, 1);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    auto mapped = MapBufferRange<const GLfloat>(GL_SHADER_STORAGE_BUFFER, 0, 16, GL_MAP_READ_BIT);
    ASSERT_FALSE(mapped.empty());

    EXPECT_GL_NO_ERROR();
    for (GLfloat value : mapped)
    {
        EXPECT_EQ(1.0, value);
    }
}

// Replicate the dEQP test dEQP-GLES31.functional.synchronization.in_invocation.ssbo_alias_overwrite
TEST_P(ComputeShaderTest, SSBOAliasOverWrite)
{
    constexpr char kCSSource[] = R"(#version 310 es
    layout (local_size_x=16, local_size_y=8) in;
    layout(binding=0, std430) buffer Output {
        highp int values[];
    } sb_result;
    layout(binding=1, std430) coherent buffer Storage0
    {
        highp int values[];
    } sb_store0;
    layout(binding=2, std430) coherent buffer Storage1
    {
        highp int values[];
    } sb_store1;

    highp int getIndex(in highp uvec2 localID, in highp int element)
    {
        highp uint groupNdx = gl_NumWorkGroups.x * gl_WorkGroupID.y + gl_WorkGroupID.x;
        return int((localID.y * gl_NumWorkGroups.x * gl_NumWorkGroups.y * gl_WorkGroupSize.x) + (groupNdx * gl_WorkGroupSize.x) + localID.x) * 8 + element;
    }

    void main (void)
    {
        int resultNdx = int(gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x + gl_GlobalInvocationID.x);
        int groupNdx = int(gl_NumWorkGroups.x * gl_WorkGroupID.y + gl_WorkGroupID.x);
        bool allOk = true;

        sb_store0.values[getIndex(gl_LocalInvocationID.xy, 0)] = 456;
        sb_store0.values[getIndex(gl_LocalInvocationID.xy, 1)] = 456;
        sb_store0.values[getIndex(gl_LocalInvocationID.xy, 2)] = 456;
        sb_store0.values[getIndex(gl_LocalInvocationID.xy, 3)] = 456;
        sb_store0.values[getIndex(gl_LocalInvocationID.xy, 4)] = 456;
        sb_store0.values[getIndex(gl_LocalInvocationID.xy, 5)] = 456;
        sb_store0.values[getIndex(gl_LocalInvocationID.xy, 6)] = 456;
        sb_store0.values[getIndex(gl_LocalInvocationID.xy, 7)] = 456;

        sb_store1.values[getIndex(gl_LocalInvocationID.xy, 0)] = groupNdx;
        sb_store1.values[getIndex(gl_LocalInvocationID.xy, 1)] = groupNdx;
        sb_store1.values[getIndex(gl_LocalInvocationID.xy, 2)] = groupNdx;
        sb_store1.values[getIndex(gl_LocalInvocationID.xy, 3)] = groupNdx;
        sb_store1.values[getIndex(gl_LocalInvocationID.xy, 4)] = groupNdx;
        sb_store1.values[getIndex(gl_LocalInvocationID.xy, 5)] = groupNdx;
        sb_store1.values[getIndex(gl_LocalInvocationID.xy, 6)] = groupNdx;
        sb_store1.values[getIndex(gl_LocalInvocationID.xy, 7)] = groupNdx;

        allOk = allOk && (sb_store0.values[getIndex(gl_LocalInvocationID.xy, 0)] == groupNdx);
        allOk = allOk && (sb_store0.values[getIndex(gl_LocalInvocationID.xy, 1)] == groupNdx);
        allOk = allOk && (sb_store0.values[getIndex(gl_LocalInvocationID.xy, 2)] == groupNdx);
        allOk = allOk && (sb_store0.values[getIndex(gl_LocalInvocationID.xy, 3)] == groupNdx);
        allOk = allOk && (sb_store0.values[getIndex(gl_LocalInvocationID.xy, 4)] == groupNdx);
        allOk = allOk && (sb_store0.values[getIndex(gl_LocalInvocationID.xy, 5)] == groupNdx);
        allOk = allOk && (sb_store0.values[getIndex(gl_LocalInvocationID.xy, 6)] == groupNdx);
        allOk = allOk && (sb_store0.values[getIndex(gl_LocalInvocationID.xy, 7)] == groupNdx);

        sb_result.values[resultNdx] = allOk ? (1) : (2);

    })";

    const int totalWorkWidth        = 256;
    const int totalWorkHeight       = 256;
    const int elementsPerInvocation = 8;

    // define compute shader input storage buffer
    const int inputSSBOBufferSizeInBytes =
        totalWorkWidth * totalWorkHeight * elementsPerInvocation * sizeof(uint32_t);
    const int inputSSBOBufferElementsCount =
        totalWorkWidth * totalWorkHeight * elementsPerInvocation;
    std::vector<uint32_t> zeros(inputSSBOBufferElementsCount, 0);
    GLBuffer ssbo;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, inputSSBOBufferSizeInBytes, zeros.data(),
                 GL_STATIC_DRAW);
    ASSERT_GL_NO_ERROR();

    // define compute shader output buffer
    const int outputBufferSizeInBytes   = totalWorkWidth * totalWorkHeight * sizeof(int32_t);
    const int outputBufferElementsCount = totalWorkWidth * totalWorkHeight;
    std::vector<int32_t> minusOnes(outputBufferElementsCount, -1);
    GLBuffer resultBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, resultBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, outputBufferSizeInBytes, &minusOnes[0], GL_STATIC_DRAW);
    ASSERT_GL_NO_ERROR();

    // dispatch compute shader
    const int localWidth  = 16;
    const int localHeight = 8;
    ASSERT(totalWorkWidth % localWidth == 0);
    ASSERT(totalWorkHeight % localHeight == 0);
    const int numGroupDimX = totalWorkWidth / localWidth;
    const int numGroupDimY = totalWorkHeight / localHeight;

    ANGLE_GL_COMPUTE_PROGRAM(csProgram, kCSSource);
    glUseProgram(csProgram);
    ASSERT_GL_NO_ERROR();

    // Bind storage buffer to compute shader binding locations
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, resultBuffer);

    glDispatchCompute(numGroupDimX, numGroupDimY, 1);
    ASSERT_GL_NO_ERROR();

    // verify the result
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, resultBuffer);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    auto mappedResults = MapBufferRange<const int32_t>(GL_SHADER_STORAGE_BUFFER, 0,
                                                       outputBufferSizeInBytes, GL_MAP_READ_BIT);
    ASSERT_FALSE(mappedResults.empty());
    std::vector<int32_t> results(outputBufferElementsCount);
    Span(results).copy_from(mappedResults);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    ASSERT_GL_NO_ERROR();

    bool error = false;
    for (int result : results)
    {
        if (result != 1)
        {
            error = true;
        }
    }
    EXPECT_EQ(false, error);
}

// Performs an atomic operation and assigns the previous value to an SSBO.
TEST_P(ComputeShaderTest, AtomicOpPreviousValueAssignedToSSBO)
{

    constexpr char kCSSource[] = R"(#version 310 es
    shared int wg;
    layout(binding = 0, std430) buffer Storage0 {
      int inner[16];
    } buf;

    layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
    void main() {
      wg = 0;
      atomicExchange(wg, 0);
      barrier();
      buf.inner[gl_WorkGroupID.x] = atomicOr(wg, 1);
    })";

    const int dispatchSize = 16;

    // define compute shader output buffer
    const int outputBufferSizeInBytes   = dispatchSize * sizeof(int32_t);
    const int outputBufferElementsCount = dispatchSize;
    std::vector<int32_t> minusOnes(outputBufferElementsCount, -1);
    GLBuffer resultBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, resultBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, outputBufferSizeInBytes, &minusOnes[0], GL_STATIC_DRAW);
    ASSERT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(csProgram, kCSSource);
    glUseProgram(csProgram);
    ASSERT_GL_NO_ERROR();

    // Bind storage buffer to compute shader binding locations
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, resultBuffer);

    glDispatchCompute(dispatchSize, 1, 1);
    ASSERT_GL_NO_ERROR();

    // verify the result
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, resultBuffer);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    auto mappedResults = MapBufferRange<const int32_t>(GL_SHADER_STORAGE_BUFFER, 0,
                                                       outputBufferSizeInBytes, GL_MAP_READ_BIT);
    ASSERT_FALSE(mappedResults.empty());
    std::vector<int32_t> results(outputBufferElementsCount);
    Span(results).copy_from(mappedResults);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    ASSERT_GL_NO_ERROR();

    for (int result : results)
    {
        EXPECT_EQ(result, 0);
    }
}

class StorageImageRenderProgramTest : public ANGLETest<>
{};

// Test creating a program with a vertex shader using storage image.
TEST_P(StorageImageRenderProgramTest, StorageImageInVertexShader)
{
    GLint maxVertexShaderImage = 0;
    glGetIntegerv(GL_MAX_VERTEX_IMAGE_UNIFORMS, &maxVertexShaderImage);

    // MAX_VERTEX_IMAGE_UNIFORMS can be 0 according to OpenGL ES 3.1 SPEC.
    ANGLE_SKIP_TEST_IF(maxVertexShaderImage == 0);

    constexpr char kVSSource_readonly[] = R"(#version 310 es
layout(rgba32f, binding = 0) readonly uniform highp image2D uIn;
void main()
{
    gl_Position = imageLoad(uIn, ivec2(0, 0));
})";

    constexpr char kVSSource_writeonly[] = R"(#version 310 es
layout(rgba32f, binding = 0) writeonly uniform highp image2D uOut;
void main()
{
    gl_Position = vec4(0, 0, 0, 1);
    imageStore(uOut, ivec2(0, 0), vec4(0, 0, 0, 1));
})";

    constexpr char kVSSource_readwrite[] = R"(#version 310 es
layout(r32f, binding = 0) uniform highp image2D uImage;
void main()
{
    gl_Position = imageLoad(uImage, ivec2(0, 0));
    imageStore(uImage, ivec2(0, 0), vec4(0, 0, 0, 1));
})";

    constexpr char kFSSource[] = R"(#version 310 es
precision mediump float;
out vec4 out_FragColor;
void main()
{
    out_FragColor = vec4(0, 1, 0, 1);
})";
    ANGLE_GL_PROGRAM(program1, kVSSource_readonly, kFSSource);
    EXPECT_GL_NO_ERROR();

    ANGLE_GL_PROGRAM(program2, kVSSource_writeonly, kFSSource);
    EXPECT_GL_NO_ERROR();

    ANGLE_GL_PROGRAM(program3, kVSSource_readwrite, kFSSource);
    EXPECT_GL_NO_ERROR();
}

// Test reading from a storage image that was bound to a deleted texture.
TEST_P(ComputeShaderTest, DeleteTextureBoundToImageUnit)
{
    constexpr char kShader[] = R"(#version 310 es
layout(local_size_x=1) in;
layout(r32f, binding=0) uniform highp readonly image2D img;
layout(std430, binding=1) buffer Out {
    float val;
} out_data;

void main() {
    out_data.val = imageLoad(img, ivec2(0,0)).x;
})";

    ANGLE_GL_COMPUTE_PROGRAM(program, kShader);
    glUseProgram(program);

    // Create a texture and fill it with a value.
    GLTexture tex;
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, 1, 1);
    float initValue = 1.0f;
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RED, GL_FLOAT, &initValue);

    // Bind it to an image unit.
    glBindImageTexture(0, tex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);

    // Create a buffer for output.
    GLBuffer buf;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buf);

    // Dispatch once to synchronize all dirty bits.
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    // Verify output is 1.0.
    {
        auto mapped = MapBufferRange<const float>(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float),
                                                  GL_MAP_READ_BIT);
        ASSERT_FALSE(mapped.empty());
        const float result = mapped[0];
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        EXPECT_EQ(1.0f, result);
    }

    // Delete the texture. This should detach it from the image unit.
    tex.reset();

    // Reset the buffer to another value and dispatch again.
    float resetValue = 0.5f;
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float), &resetValue);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    // Verify output is 0.0 (null image).
    {
        auto mapped = MapBufferRange<const float>(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float),
                                                  GL_MAP_READ_BIT);
        ASSERT_FALSE(mapped.empty());
        const float result = mapped[0];
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        EXPECT_EQ(0.0f, result);
    }

    EXPECT_GL_NO_ERROR();
}

TEST_P(ExperimentalGles31MetalComputeTest, TextureBufferSamplerAndImageFormatReinterpretation)
{
    ANGLE_SKIP_TEST_IF(!IsGLExtensionEnabled("GL_EXT_texture_buffer"));

    constexpr char kComputeShader[] = R"(#version 310 es
#extension GL_EXT_texture_buffer : require
layout(local_size_x = 1) in;
layout(rgba8ui, binding = 0) writeonly uniform highp uimageBuffer image;
void main()
{
    if (imageSize(image) >= 4)
    {
        imageStore(image, 3, uvec4(255u, 0u, 0u, 255u));
    }
})";
    constexpr char kFragmentShader[] = R"(#version 310 es
#extension GL_EXT_texture_buffer : require
precision highp float;
uniform highp samplerBuffer source;
out vec4 colorOut;
void main()
{
    colorOut = texelFetch(source, 3);
})";

    const std::array<GLubyte, 16> initialData = {0,   255, 0,   255, 0, 0, 255, 255,
                                                  255, 0,   0,   255, 0, 255, 0,   255};
    GLBuffer buffer;
    glBindBuffer(GL_TEXTURE_BUFFER, buffer);
    glBufferData(GL_TEXTURE_BUFFER, initialData.size(), initialData.data(), GL_DYNAMIC_DRAW);

    GLTexture texture;
    glBindTexture(GL_TEXTURE_BUFFER, texture);
    glTexBufferEXT(GL_TEXTURE_BUFFER, GL_RGBA8, buffer);
    ASSERT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(computeProgram, kComputeShader);
    glUseProgram(computeProgram);
    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8UI);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
    ASSERT_GL_NO_ERROR();

    ANGLE_GL_PROGRAM(program, essl31_shaders::vs::Simple(), kFragmentShader);
    glUseProgram(program);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER, texture);
    glUniform1i(glGetUniformLocation(program, "source"), 0);
    drawQuad(program, essl31_shaders::PositionAttrib(), 0.5);
    ASSERT_GL_NO_ERROR();
    EXPECT_PIXEL_COLOR_EQ(0, 0, GLColor::red);
}

TEST_P(ExperimentalGles31MetalComputeTest, TextureBufferImageAtomicAdd)
{
    ANGLE_SKIP_TEST_IF(!IsGLExtensionEnabled("GL_EXT_texture_buffer"));
    ANGLE_SKIP_TEST_IF(!IsGLExtensionEnabled("GL_OES_shader_image_atomic"));

    constexpr char kComputeShader[] = R"(#version 310 es
#extension GL_EXT_texture_buffer : require
#extension GL_OES_shader_image_atomic : require
layout(local_size_x = 1) in;
layout(r32ui, binding = 0) uniform highp uimageBuffer image;
void main()
{
    imageAtomicAdd(image, 0, 3u);
})";

    const GLuint initialValue = 4;
    GLBuffer buffer;
    glBindBuffer(GL_TEXTURE_BUFFER, buffer);
    glBufferData(GL_TEXTURE_BUFFER, sizeof(initialValue), &initialValue, GL_DYNAMIC_DRAW);

    GLTexture texture;
    glBindTexture(GL_TEXTURE_BUFFER, texture);
    glTexBufferEXT(GL_TEXTURE_BUFFER, GL_R32UI, buffer);
    ASSERT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(program, kComputeShader);
    glUseProgram(program);
    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    ASSERT_GL_NO_ERROR();

    const auto *result = static_cast<const GLuint *>(
        glMapBufferRange(GL_TEXTURE_BUFFER, 0, sizeof(initialValue), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, result);
    EXPECT_EQ(7u, *result);
    EXPECT_GL_TRUE(glUnmapBuffer(GL_TEXTURE_BUFFER));
    EXPECT_GL_NO_ERROR();
}

TEST_P(ExperimentalGles31MetalComputeTest, TextureBufferR32fImageAtomicExchange)
{
    ANGLE_SKIP_TEST_IF(!IsGLExtensionEnabled("GL_EXT_texture_buffer"));
    ANGLE_SKIP_TEST_IF(!IsGLExtensionEnabled("GL_OES_shader_image_atomic"));

    constexpr char kComputeShader[] = R"(#version 310 es
#extension GL_EXT_texture_buffer : require
#extension GL_OES_shader_image_atomic : require
layout(local_size_x = 1) in;
layout(r32f, binding = 0) uniform highp imageBuffer floatImage;
layout(std430, binding = 0) buffer Output
{
    vec2 values;
} outputData;
void main()
{
    outputData.values.x = imageAtomicExchange(floatImage, 0, 2.5);
    outputData.values.y = imageLoad(floatImage, 0).r;
})";

    constexpr GLfloat initialValue = 1.25f;
    GLBuffer textureBuffer;
    glBindBuffer(GL_TEXTURE_BUFFER, textureBuffer);
    glBufferData(GL_TEXTURE_BUFFER, sizeof(initialValue), &initialValue, GL_DYNAMIC_DRAW);

    GLTexture floatImage;
    glBindTexture(GL_TEXTURE_BUFFER, floatImage);
    glTexBufferEXT(GL_TEXTURE_BUFFER, GL_R32F, textureBuffer);
    ASSERT_GL_NO_ERROR();

    ANGLE_GL_COMPUTE_PROGRAM(program, kComputeShader);
    GLBuffer outputBuffer;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 2 * sizeof(GLfloat), nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, outputBuffer);
    glBindImageTexture(0, floatImage, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F);
    glUseProgram(program);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    ASSERT_GL_NO_ERROR();

    const auto *actual = static_cast<const GLfloat *>(
        glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 2 * sizeof(GLfloat), GL_MAP_READ_BIT));
    ASSERT_NE(nullptr, actual);
    EXPECT_FLOAT_EQ(initialValue, actual[0]);
    EXPECT_FLOAT_EQ(2.5f, actual[1]);
    EXPECT_GL_TRUE(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));
    EXPECT_GL_NO_ERROR();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ComputeShaderTest);
ANGLE_INSTANTIATE_TEST_ES31_AND(ComputeShaderTest,
                                ES31_VULKAN().enable(Feature::ForceRobustResourceInit),
                                ES31_VULKAN().enable(Feature::AvoidRelaxedPrecisionOnIntegers));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ComputeShaderTestES3);
ANGLE_INSTANTIATE_TEST_ES3(ComputeShaderTestES3);

ANGLE_INSTANTIATE_TEST(ExperimentalGles31MetalComputeTest, ExperimentalGles31Metal());

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(StorageImageRenderProgramTest);
ANGLE_INSTANTIATE_TEST_ES31(StorageImageRenderProgramTest);
}  // namespace
