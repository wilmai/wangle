//
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include <algorithm>
#include <cctype>
#include <map>
#include <string>
#include <unordered_set>

#include "common/system_utils.h"
#include "common/unsafe_buffers.h"
#include "compiler/translator/BaseTypes.h"
#include "compiler/translator/ImmutableStringBuilder.h"
#include "compiler/translator/Name.h"
#include "compiler/translator/OutputTree.h"
#include "compiler/translator/SymbolTable.h"
#include "compiler/translator/msl/AstHelpers.h"
#include "compiler/translator/msl/DebugSink.h"
#include "compiler/translator/msl/EmitMetal.h"
#include "compiler/translator/msl/Layout.h"
#include "compiler/translator/msl/ProgramPrelude.h"
#include "compiler/translator/msl/RewritePipelines.h"
#include "compiler/translator/msl/TranslatorMSL.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

using namespace sh;

////////////////////////////////////////////////////////////////////////////////

#if defined(ANGLE_ENABLE_ASSERTS)
using Sink = DebugSink;
#else
using Sink = TInfoSinkBase;
#endif

////////////////////////////////////////////////////////////////////////////////

namespace
{

struct VarDecl
{
    explicit VarDecl(const TVariable &var) : mVariable(&var), mIsField(false) {}
    explicit VarDecl(const TField &field) : mField(&field), mIsField(true) {}

    ANGLE_INLINE const TVariable &variable() const
    {
        ASSERT(isVariable());
        return *mVariable;
    }

    ANGLE_INLINE const TField &field() const
    {
        ASSERT(isField());
        return *mField;
    }

    ANGLE_INLINE bool isVariable() const { return !mIsField; }

    ANGLE_INLINE bool isField() const { return mIsField; }

    const TType &type() const { return isField() ? *field().type() : variable().getType(); }

    SymbolType symbolType() const
    {
        return isField() ? field().symbolType() : variable().symbolType();
    }

  private:
    union
    {
        const TVariable *mVariable;
        const TField *mField;
    };
    bool mIsField;
};

constexpr AddressSpace kThreadgroupAddressSpace = AddressSpace::Threadgroup;
// Reserve one range for all sampler and image bindings before placing the extra 3D texture views
// used to emulate 2D images backed by 3D textures.
constexpr size_t kMetalImage3DBindingOffset = 24;  // 8 image units + 16 sampler units.

Layout GetGlslStorageBufferLayoutOf(const TType &type,
                                    TLayoutBlockStorage storage,
                                    TLayoutMatrixPacking packing);

bool GetStorageBufferArrayWrapperStride(
    const TType &type,
    size_t *strideOut,
    TLayoutBlockStorage inheritedStorage = TLayoutBlockStorage::EbsStd430)
{
    if (!type.isArray() || type.getNumArraySizes() != 1 || type.isUnsizedArray())
    {
        return false;
    }

    TType elementType(type);
    elementType.toArrayElementType();
    if (!(elementType.isScalar() || elementType.isVector()) ||
        (elementType.getBasicType() != TBasicType::EbtFloat &&
         elementType.getBasicType() != TBasicType::EbtInt &&
         elementType.getBasicType() != TBasicType::EbtUInt))
    {
        return false;
    }

    const TLayoutBlockStorage storage = Overlay(inheritedStorage, type);
    const Layout glslArray =
        GetGlslStorageBufferLayoutOf(type, storage, TLayoutMatrixPacking::EmpColumnMajor);
    const size_t glslStride = glslArray.sizeOf / type.getOutermostArraySize();
    const size_t metalSize  = MetalLayoutOf(elementType).sizeOf;
    if (glslStride <= metalSize)
    {
        return false;
    }

    *strideOut = glslStride;
    return true;
}

std::string GetStorageBufferArrayWrapperName(const TStructure &parent,
                                             const TField &field,
                                             TLayoutBlockStorage storage)
{
    return "ANGLE_SSBO_array_" + std::to_string(parent.uniqueId().get()) + "_" +
           field.name().data() + "_" + std::to_string(static_cast<int>(storage));
}

bool IsStorageBufferMatrixField(const TType &type)
{
    TType baseType(type);
    baseType.toArrayBaseType();
    return baseType.isMatrix();
}

std::string GetStorageBufferMatrixWrapperName(const TStructure &parent, const TField &field)
{
    return "ANGLE_SSBO_matrix_" + std::to_string(parent.uniqueId().get()) + "_" +
           field.name().data();
}

void GetStorageBufferMatrixLayout(const TType &matrixType,
                                  size_t *majorCountOut,
                                  size_t *vectorSizeOut,
                                  size_t *vectorStrideOut,
                                  bool *isRowMajorOut)
{
    const TLayoutBlockStorage storage  = Overlay(TLayoutBlockStorage::EbsStd430, matrixType);
    const TLayoutMatrixPacking packing = Overlay(TLayoutMatrixPacking::EmpColumnMajor, matrixType);
    const bool isRowMajor              = packing == TLayoutMatrixPacking::EmpRowMajor;
    const size_t majorCount            = isRowMajor ? matrixType.getRows() : matrixType.getCols();
    const size_t vectorSize            = isRowMajor ? matrixType.getCols() : matrixType.getRows();
    const Layout vectorLayout =
        GetGlslStorageBufferLayoutOf(TType(TBasicType::EbtFloat, static_cast<uint8_t>(vectorSize)),
                                     storage, TLayoutMatrixPacking::EmpColumnMajor);
    size_t vectorAlignment = vectorLayout.alignOf;
    if (storage == TLayoutBlockStorage::EbsStd140 || storage == TLayoutBlockStorage::EbsShared)
    {
        vectorAlignment = std::max<size_t>(vectorAlignment, 16);
    }
    const size_t vectorStride =
        ((vectorLayout.sizeOf + vectorAlignment - 1) / vectorAlignment) * vectorAlignment;
    *majorCountOut   = majorCount;
    *vectorSizeOut   = vectorSize;
    *vectorStrideOut = vectorStride;
    *isRowMajorOut   = isRowMajor;
}

size_t RoundUpToStorageAlignment(size_t size, size_t alignment)
{
    return alignment == 0 ? size : ((size + alignment - 1) / alignment) * alignment;
}

Layout GetGlslStorageBufferLayoutOf(const TType &type,
                                    TLayoutBlockStorage storage,
                                    TLayoutMatrixPacking packing)
{
    storage = Overlay(storage, type);
    packing = Overlay(packing, type);

    if (type.isArray())
    {
        TType elementType(type);
        elementType.toArrayElementType();
        Layout elementLayout  = GetGlslStorageBufferLayoutOf(elementType, storage, packing);
        size_t arrayAlignment = elementLayout.alignOf;
        if (storage == TLayoutBlockStorage::EbsStd140 || storage == TLayoutBlockStorage::EbsShared)
        {
            arrayAlignment = std::max<size_t>(arrayAlignment, 16);
        }
        const size_t stride = RoundUpToStorageAlignment(elementLayout.sizeOf, arrayAlignment);
        return {stride * type.getOutermostArraySize(), arrayAlignment};
    }

    if (const TStructure *structure = type.getStruct())
    {
        size_t offset    = 0;
        size_t alignment = 1;
        for (const TField *field : structure->fields())
        {
            const Layout fieldLayout =
                GetGlslStorageBufferLayoutOf(*field->type(), storage, packing);
            offset = RoundUpToStorageAlignment(offset, fieldLayout.alignOf);
            offset += fieldLayout.sizeOf;
            alignment = std::max(alignment, fieldLayout.alignOf);
        }
        if (storage == TLayoutBlockStorage::EbsStd140 || storage == TLayoutBlockStorage::EbsShared)
        {
            alignment = std::max<size_t>(alignment, 16);
        }
        return {RoundUpToStorageAlignment(offset, alignment), alignment};
    }

    if (type.isMatrix())
    {
        size_t majorCount;
        size_t vectorSize;
        size_t vectorStride;
        bool isRowMajor;
        GetStorageBufferMatrixLayout(type, &majorCount, &vectorSize, &vectorStride, &isRowMajor);
        ASSERT(vectorSize == (isRowMajor ? type.getCols() : type.getRows()));
        return {majorCount * vectorStride, vectorStride};
    }

    if (type.isVector())
    {
        const size_t vectorSize    = type.getNominalSize();
        const size_t componentSize = type.getBasicType() == TBasicType::EbtBool ? 4 : 4;
        const size_t alignment     = vectorSize == 1   ? componentSize
                                     : vectorSize == 2 ? componentSize * 2
                                                       : componentSize * 4;
        return {componentSize * vectorSize, alignment};
    }

    return {4, 4};
}

bool NeedsStorageBufferMatrixWrapper(const TType &type)
{
    if (!IsStorageBufferMatrixField(type))
    {
        return false;
    }

    TType matrixType(type);
    matrixType.toArrayBaseType();
    size_t majorCount;
    size_t vectorSize;
    size_t vectorStride;
    bool isRowMajor;
    GetStorageBufferMatrixLayout(matrixType, &majorCount, &vectorSize, &vectorStride, &isRowMajor);
    if (isRowMajor)
    {
        return true;
    }
    return majorCount * vectorStride != MetalLayoutOf(matrixType).sizeOf;
}

bool NeedsStorageBufferStructVariant(const TStructure &structure,
                                     TLayoutBlockStorage storage,
                                     TLayoutMatrixPacking packing)
{
    size_t glslOffset          = 0;
    size_t metalOffset         = 0;
    size_t glslAlignment       = 1;
    size_t metalAlignment      = 1;
    bool needsNestedConversion = false;

    for (const TField *field : structure.fields())
    {
        const TType &fieldType                  = *field->type();
        const TLayoutBlockStorage fieldStorage  = Overlay(storage, fieldType);
        const TLayoutMatrixPacking fieldPacking = Overlay(packing, fieldType);
        const Layout glslFieldLayout =
            GetGlslStorageBufferLayoutOf(fieldType, fieldStorage, fieldPacking);
        const Layout metalFieldLayout = MetalLayoutOf(fieldType);

        glslOffset  = RoundUpToStorageAlignment(glslOffset, glslFieldLayout.alignOf);
        metalOffset = RoundUpToStorageAlignment(metalOffset, metalFieldLayout.alignOf);
        if (glslOffset != metalOffset)
        {
            return true;
        }
        glslOffset += glslFieldLayout.sizeOf;
        metalOffset += metalFieldLayout.sizeOf;
        glslAlignment  = std::max(glslAlignment, glslFieldLayout.alignOf);
        metalAlignment = std::max(metalAlignment, metalFieldLayout.alignOf);

        TType baseType(fieldType);
        baseType.toArrayBaseType();
        if (const TStructure *nested = baseType.getStruct())
        {
            needsNestedConversion |=
                NeedsStorageBufferStructVariant(*nested, fieldStorage, fieldPacking);
        }
        if (IsStorageBufferMatrixField(fieldType) && NeedsStorageBufferMatrixWrapper(fieldType))
        {
            needsNestedConversion = true;
        }
    }

    if (storage == TLayoutBlockStorage::EbsStd140 || storage == TLayoutBlockStorage::EbsShared)
    {
        glslAlignment = std::max<size_t>(glslAlignment, 16);
    }
    const size_t glslSize  = RoundUpToStorageAlignment(glslOffset, glslAlignment);
    const size_t metalSize = RoundUpToStorageAlignment(metalOffset, metalAlignment);
    return needsNestedConversion || glslSize != metalSize || glslAlignment != metalAlignment;
}

class GenMetalTraverser : public TIntermTraverser
{
  public:
    ~GenMetalTraverser() override;

    GenMetalTraverser(const TCompiler &compiler,
                      Sink &out,
                      IdGen &idGen,
                      const PipelineStructs &pipelineStructs,
                      SymbolEnv &symbolEnv,
                      const ProgramPreludeConfig &ppc,
                      const ShCompileOptions &compileOptions);

    void visitSymbol(TIntermSymbol *) override;
    void visitConstantUnion(TIntermConstantUnion *) override;
    bool visitSwizzle(Visit, TIntermSwizzle *) override;
    bool visitBinary(Visit, TIntermBinary *) override;
    bool visitUnary(Visit, TIntermUnary *) override;
    bool visitTernary(Visit, TIntermTernary *) override;
    bool visitIfElse(Visit, TIntermIfElse *) override;
    bool visitSwitch(Visit, TIntermSwitch *) override;
    bool visitCase(Visit, TIntermCase *) override;
    void visitFunctionPrototype(TIntermFunctionPrototype *) override;
    bool visitFunctionDefinition(Visit, TIntermFunctionDefinition *) override;
    bool visitAggregate(Visit, TIntermAggregate *) override;
    bool visitBlock(Visit, TIntermBlock *) override;
    bool visitGlobalQualifierDeclaration(Visit, TIntermGlobalQualifierDeclaration *) override;
    bool visitDeclaration(Visit, TIntermDeclaration *) override;
    bool visitLoop(Visit, TIntermLoop *) override;
    bool visitForLoop(TIntermLoop *);
    bool visitWhileLoop(TIntermLoop *);
    bool visitDoWhileLoop(TIntermLoop *);
    bool visitBranch(Visit, TIntermBranch *) override;

  private:
    using FuncToName = std::map<ImmutableString, Name>;
    static FuncToName BuildFuncToName();

    struct EmitVariableDeclarationConfig
    {
        bool isParameter                        = false;
        bool isMainParameter                    = false;
        bool emitPostQualifier                  = false;
        bool isPacked                           = false;
        bool disableStructSpecifier             = false;
        bool isUBO                              = false;
        const AddressSpace *isPointer           = nullptr;
        const AddressSpace *isReference         = nullptr;
        const AddressSpace *storageAddressSpace = nullptr;
    };

    struct EmitTypeConfig
    {
        const EmitVariableDeclarationConfig *evdConfig = nullptr;
    };

    void emitIndentation();
    void emitOpeningPointerParen();
    void emitClosingPointerParen();
    void emitFunctionSignature(const TFunction &func);
    void emitFunctionReturn(const TFunction &func);
    void emitFunctionParameter(const TFunction &func, const TVariable &param);
    void emitStorageBufferMatrixHelpers();
    void emitStorageBufferArrayWrapper(const TStructure &parent,
                                       const TField &field,
                                       TLayoutBlockStorage storage);
    std::string getStorageBufferStructVariantName(const TStructure &structure,
                                                  TLayoutBlockStorage storage,
                                                  TLayoutMatrixPacking packing) const;
    void emitStorageBufferStructVariant(const TStructure &structure,
                                        TLayoutBlockStorage storage,
                                        TLayoutMatrixPacking packing);
    void emitStorageBufferPadding(size_t byteCount, size_t padIndex);

    void emitNameOf(const TField &object);
    void emitNameOf(const TSymbol &object);
    void emitNameOf(const TVariable &object);
    void emitBlockNameOf(const TSymbol &object);
    void emitNameOf(const VarDecl &object);

    void emitBareTypeName(const TType &type, const EmitTypeConfig &etConfig);
    void emitType(const TType &type, const EmitTypeConfig &etConfig);
    void emitPostQualifier(const EmitVariableDeclarationConfig &evdConfig,
                           const VarDecl &decl,
                           const TQualifier qualifier);

    void emitLoopBody(TIntermBlock *bodyNode);

    struct FieldAnnotationIndices
    {
        size_t attribute = 0;
        size_t color     = 0;
    };

    void emitFieldDeclaration(const TField &field,
                              const TStructure &parent,
                              FieldAnnotationIndices &annotationIndices);
    void emitAttributeDeclaration(const TField &field, FieldAnnotationIndices &annotationIndices);
    void emitUniformBufferDeclaration(const TField &field,
                                      FieldAnnotationIndices &annotationIndices);
    void emitStorageBufferDeclaration(const TField &field,
                                      FieldAnnotationIndices &annotationIndices);
    void emitStorageBufferLengthDeclaration(const TField &field,
                                            FieldAnnotationIndices &annotationIndices);
    void emitStructDeclaration(const TType &type);
    void emitOrdinaryVariableDeclaration(const VarDecl &decl,
                                         const EmitVariableDeclarationConfig &evdConfig);
    void emitVariableDeclaration(const VarDecl &decl,
                                 const EmitVariableDeclarationConfig &evdConfig);

    void emitOpenBrace();
    void emitCloseBrace();

    void groupedTraverse(TIntermNode &node);

    const TField &getDirectField(const TFieldListCollection &fieldsNode,
                                 const TConstantUnion &index);
    const TField &getDirectField(const TIntermTyped &fieldsNode, TIntermTyped &indexNode);
    const TField *getStorageBufferField(TIntermTyped &expression,
                                        const TStructure **parentOut = nullptr);
    const TField *getStorageBufferArrayField(TIntermTyped &expression);

    const TConstantUnion *emitConstantUnionArray(const TConstantUnion *const constUnion,
                                                 const size_t size);

    const TConstantUnion *emitConstantUnion(const TType &type, const TConstantUnion *constUnion);

    void emitSingleConstant(const TConstantUnion *const constUnion);

  private:
    Sink &mOut;
    const TCompiler &mCompiler;
    const PipelineStructs &mPipelineStructs;
    SymbolEnv &mSymbolEnv;
    IdGen &mIdGen;
    const ShCompileOptions &mCompileOptions;
    int mIndentLevel                  = -1;
    int mLastIndentationPos           = -1;
    int mOpenPointerParenCount        = 0;
    bool mParentIsSwitch              = false;
    bool isTraversingVertexMain       = false;
    bool mIsComputeMain               = false;
    bool mInjectSharedVariables       = false;
    bool mTemporarilyDisableSemicolon = false;
    std::vector<const TVariable *> mSharedVariables;
    std::unordered_map<const TSymbol *, Name> mRenamedSymbols;
    const FuncToName mFuncToName            = BuildFuncToName();
    size_t mMainTextureIndex                = 0;
    size_t mMainSamplerIndex                = 0;
    size_t mMainUniformBufferIndex          = 0;
    size_t mDriverUniformsBindingIndex      = 0;
    size_t mUBOArgumentBufferBindingIndex   = 0;
    size_t mSSBOArgumentBufferBindingIndex  = 0;
    bool mRasterOrderGroupsSupported        = false;
    bool mWriteHelperSampleMask             = false;
    bool mSuppressStorageBufferMatrixLoad   = false;
    bool mStorageBufferMatrixHelpersEmitted = false;
    std::vector<std::pair<const TStructure *, const TField *>> mStorageBufferMatrixFields;
    std::unordered_set<std::string> mEmittedStorageBufferArrayWrappers;
    std::unordered_set<std::string> mEmittedStorageBufferStructVariants;
};

bool IsImage2DType(TBasicType type)
{
    return type == EbtImage2D || type == EbtIImage2D || type == EbtUImage2D;
}

bool IsImageCubeType(TBasicType type)
{
    return type == EbtImageCube || type == EbtIImageCube || type == EbtUImageCube;
}

const char *GetImageAtomicFunctionName(TOperator op)
{
    switch (op)
    {
        case TOperator::EOpImageAtomicAdd:
            return "ANGLE_imageAtomicAdd";
        case TOperator::EOpImageAtomicMin:
            return "ANGLE_imageAtomicMin";
        case TOperator::EOpImageAtomicMax:
            return "ANGLE_imageAtomicMax";
        case TOperator::EOpImageAtomicAnd:
            return "ANGLE_imageAtomicAnd";
        case TOperator::EOpImageAtomicOr:
            return "ANGLE_imageAtomicOr";
        case TOperator::EOpImageAtomicXor:
            return "ANGLE_imageAtomicXor";
        case TOperator::EOpImageAtomicExchange:
            return "ANGLE_imageAtomicExchange";
        case TOperator::EOpImageAtomicCompSwap:
            return "ANGLE_imageAtomicCompSwap";
        default:
            UNREACHABLE();
            return "";
    }
}

TBasicType GetImage3DType(TBasicType type)
{
    switch (type)
    {
        case EbtImage2D:
            return EbtImage3D;
        case EbtIImage2D:
            return EbtIImage3D;
        case EbtUImage2D:
            return EbtUImage3D;
        default:
            UNREACHABLE();
            return EbtImage3D;
    }
}

void EmitMetalArrayTypePrefix(Sink &out, const angle::Span<const unsigned int> &arraySizes)
{
    for (size_t index = arraySizes.size(); index > 0; --index)
    {
        out << "metal::array<";
    }
}

void EmitMetalArrayTypeSuffix(Sink &out,
                              const angle::Span<const unsigned int> &arraySizes,
                              bool emitUnsizedArrayAsOne = false)
{
    for (unsigned int size : arraySizes)
    {
        out << ", " << (emitUnsizedArrayAsOne && size == 0 ? 1 : size) << ">";
    }
}
}  // anonymous namespace

GenMetalTraverser::~GenMetalTraverser()
{
    ASSERT(mIndentLevel == -1);
    ASSERT(!mParentIsSwitch);
    ASSERT(mOpenPointerParenCount == 0);
}

GenMetalTraverser::GenMetalTraverser(const TCompiler &compiler,
                                     Sink &out,
                                     IdGen &idGen,
                                     const PipelineStructs &pipelineStructs,
                                     SymbolEnv &symbolEnv,
                                     const ProgramPreludeConfig &ppc,
                                     const ShCompileOptions &compileOptions)
    : TIntermTraverser(true, false, false),
      mOut(out),
      mCompiler(compiler),
      mPipelineStructs(pipelineStructs),
      mSymbolEnv(symbolEnv),
      mIdGen(idGen),
      mCompileOptions(compileOptions),
      mMainUniformBufferIndex(compileOptions.metal.defaultUniformsBindingIndex),
      mDriverUniformsBindingIndex(compileOptions.metal.driverUniformsBindingIndex),
      mUBOArgumentBufferBindingIndex(compileOptions.metal.UBOArgumentBufferBindingIndex),
      mSSBOArgumentBufferBindingIndex(compileOptions.metal.SSBOArgumentBufferBindingIndex),
      mRasterOrderGroupsSupported(compileOptions.pls.fragmentSyncType ==
                                  ShFragmentSynchronizationType::RasterOrderGroups_Metal),
      mWriteHelperSampleMask(ppc.writeHelperSampleMask)
{}

void GenMetalTraverser::emitIndentation()
{
    ASSERT(mIndentLevel >= 0);

    if (mLastIndentationPos == mOut.size())
    {
        return;  // Line is already indented.
    }

    for (int i = 0; i < mIndentLevel; ++i)
    {
        mOut << "  ";
    }

    mLastIndentationPos = mOut.size();
}

void GenMetalTraverser::emitOpeningPointerParen()
{
    mOut << "(*";
    mOpenPointerParenCount++;
}

void GenMetalTraverser::emitClosingPointerParen()
{
    if (mOpenPointerParenCount > 0)
    {
        mOut << ")";
        mOpenPointerParenCount--;
    }
}

static const char *GetOperatorString(TOperator op,
                                     const TType &resultType,
                                     const TType *argType0,
                                     const TType *argType1,
                                     const TType *argType2)
{
    switch (op)
    {
        case TOperator::EOpComma:
            return ",";
        case TOperator::EOpAssign:
            return "=";
        case TOperator::EOpInitialize:
            return "=";
        case TOperator::EOpAddAssign:
            return resultType.isSignedInt() ? "ANGLE_addInt" : "+=";
        case TOperator::EOpSubAssign:
            return resultType.isSignedInt() ? "ANGLE_subInt" : "-=";
        case TOperator::EOpMulAssign:
            return resultType.isSignedInt() ? "ANGLE_imul" : "*=";
        case TOperator::EOpDivAssign:
            return IsInteger(resultType.getBasicType()) ? "ANGLE_div" : "/=";
        case TOperator::EOpIModAssign:
            return "ANGLE_imod";
        case TOperator::EOpBitShiftLeftAssign:
            return resultType.isSignedInt() ? "ANGLE_ilshift" : "ANGLE_ulshift";
        case TOperator::EOpBitShiftRightAssign:
            return "ANGLE_rshift";
        case TOperator::EOpBitwiseAndAssign:
            return "&=";
        case TOperator::EOpBitwiseXorAssign:
            return "^=";
        case TOperator::EOpBitwiseOrAssign:
            return "|=";
        case TOperator::EOpAdd:
            return resultType.isSignedInt() ? "ANGLE_addInt" : "+";
        case TOperator::EOpSub:
            return resultType.isSignedInt() ? "ANGLE_subInt" : "-";
        case TOperator::EOpMul:
            return resultType.isSignedInt() ? "ANGLE_imul" : "*";
        case TOperator::EOpDiv:
            return IsInteger(resultType.getBasicType()) ? "ANGLE_div" : "/";
        case TOperator::EOpIMod:
            return "ANGLE_imod";
        case TOperator::EOpBitShiftLeft:
            return resultType.isSignedInt() ? "ANGLE_ilshift" : "ANGLE_ulshift";
        case TOperator::EOpBitShiftRight:
            return "ANGLE_rshift";
        case TOperator::EOpBitwiseAnd:
            return "&";
        case TOperator::EOpBitwiseXor:
            return "^";
        case TOperator::EOpBitwiseOr:
            return "|";
        case TOperator::EOpLessThan:
            return "<";
        case TOperator::EOpGreaterThan:
            return ">";
        case TOperator::EOpLessThanEqual:
            return "<=";
        case TOperator::EOpGreaterThanEqual:
            return ">=";
        case TOperator::EOpLessThanComponentWise:
            return "<";
        case TOperator::EOpLessThanEqualComponentWise:
            return "<=";
        case TOperator::EOpGreaterThanEqualComponentWise:
            return ">=";
        case TOperator::EOpGreaterThanComponentWise:
            return ">";
        case TOperator::EOpLogicalOr:
            return "||";
        case TOperator::EOpLogicalXor:
            return "!=/*xor*/";  // XXX: This might need to be handled differently for some obtuse
                                 // use case.
        case TOperator::EOpLogicalAnd:
            return "&&";
        case TOperator::EOpNegative:
            // Matrices have an operator- overload in the program prelude.
            if (argType0->isMatrix())
            {
                return "-";
            }
            return resultType.isSignedInt() ? "ANGLE_negInt" : "-";
        case TOperator::EOpPositive:
            if (argType0->isMatrix())
            {
                return "";
            }
            return "+";
        case TOperator::EOpLogicalNot:
            return "!";
        case TOperator::EOpNotComponentWise:
            return "!";
        case TOperator::EOpBitwiseNot:
            return "~";
        case TOperator::EOpPostIncrement:
            return resultType.isSignedInt() ? "ANGLE_postIncrementInt" : "++";
        case TOperator::EOpPostDecrement:
            return resultType.isSignedInt() ? "ANGLE_postDecrementInt" : "--";
        case TOperator::EOpPreIncrement:
            return resultType.isSignedInt() ? "ANGLE_preIncrementInt" : "++";
        case TOperator::EOpPreDecrement:
            return resultType.isSignedInt() ? "ANGLE_preDecrementInt" : "--";
        case TOperator::EOpVectorTimesScalarAssign:
            return resultType.isSignedInt() ? "ANGLE_imul" : "*=";
        case TOperator::EOpVectorTimesMatrixAssign:
            return "*=";
        case TOperator::EOpMatrixTimesScalarAssign:
            return "*=";
        case TOperator::EOpMatrixTimesMatrixAssign:
            return "*=";
        case TOperator::EOpVectorTimesScalar:
            return resultType.isSignedInt() ? "ANGLE_imul" : "*";
        case TOperator::EOpVectorTimesMatrix:
            return "*";
        case TOperator::EOpMatrixTimesVector:
            return "*";
        case TOperator::EOpMatrixTimesScalar:
            return "*";
        case TOperator::EOpMatrixTimesMatrix:
            return "*";
        case TOperator::EOpEqualComponentWise:
            return "==";
        case TOperator::EOpNotEqualComponentWise:
            return "!=";

        case TOperator::EOpEqual:
            if ((argType0->getStruct() && argType1->getStruct()) &&
                (argType0->isArray() && argType1->isArray()))
            {
                return "ANGLE_equalStructArray";
            }

            if ((argType0->isVector() && argType1->isVector()) ||
                (argType0->getStruct() && argType1->getStruct()) ||
                (argType0->isArray() && argType1->isArray()) ||
                (argType0->isMatrix() && argType1->isMatrix()))

            {
                return "ANGLE_equal";
            }

            return "==";

        case TOperator::EOpNotEqual:
            if ((argType0->getStruct() && argType1->getStruct()) &&
                (argType0->isArray() && argType1->isArray()))
            {
                return "ANGLE_notEqualStructArray";
            }

            if ((argType0->isVector() && argType1->isVector()) ||
                (argType0->isArray() && argType1->isArray()) ||
                (argType0->isMatrix() && argType1->isMatrix()))
            {
                return "ANGLE_notEqual";
            }
            else if (argType0->getStruct() && argType1->getStruct())
            {
                return "ANGLE_notEqualStruct";
            }
            return "!=";

        case TOperator::EOpKill:
            UNIMPLEMENTED();
            return "kill";
        case TOperator::EOpReturn:
            return "return";
        case TOperator::EOpBreak:
            return "break";
        case TOperator::EOpContinue:
            return "continue";

        case TOperator::EOpRadians:
            return "ANGLE_radians";
        case TOperator::EOpDegrees:
            return "ANGLE_degrees";
        case TOperator::EOpAtan:
            return argType1 == nullptr ? "metal::atan" : "metal::atan2";
        case TOperator::EOpMod:
            return "ANGLE_mod";  // differs from metal::mod
        case TOperator::EOpRefract:
            return argType0->isVector() ? "metal::refract" : "ANGLE_refract_scalar";
        case TOperator::EOpDistance:
            return argType0->isVector() ? "metal::distance" : "ANGLE_distance_scalar";
        case TOperator::EOpLength:
            return argType0->isVector() ? "metal::length" : "metal::abs";
        case TOperator::EOpDot:
            return argType0->isVector() ? "metal::dot" : "*";
        case TOperator::EOpNormalize:
            return argType0->isVector() ? "metal::fast::normalize" : "metal::sign";
        case TOperator::EOpFaceforward:
            return argType0->isVector() ? "metal::faceforward" : "ANGLE_faceforward_scalar";
        case TOperator::EOpReflect:
            return argType0->isVector() ? "metal::reflect" : "ANGLE_reflect_scalar";
        case TOperator::EOpMatrixCompMult:
            return "ANGLE_componentWiseMultiply";
        case TOperator::EOpOuterProduct:
            return "ANGLE_outerProduct";
        case TOperator::EOpSign:
            return argType0->getBasicType() == EbtFloat ? "metal::sign" : "ANGLE_sign_int";

        case TOperator::EOpAbs:
            return "metal::abs";
        case TOperator::EOpAll:
            return "metal::all";
        case TOperator::EOpAny:
            return "metal::any";
        case TOperator::EOpSin:
            return "metal::sin";
        case TOperator::EOpCos:
            return "metal::cos";
        case TOperator::EOpTan:
            return "metal::tan";
        case TOperator::EOpAsin:
            return "metal::asin";
        case TOperator::EOpAcos:
            return "metal::acos";
        case TOperator::EOpSinh:
            return "metal::sinh";
        case TOperator::EOpCosh:
            return "metal::cosh";
        case TOperator::EOpTanh:
            return resultType.getPrecision() == TPrecision::EbpHigh ? "metal::precise::tanh"
                                                                    : "metal::tanh";
        case TOperator::EOpAsinh:
            return "metal::asinh";
        case TOperator::EOpAcosh:
            return "metal::acosh";
        case TOperator::EOpAtanh:
            return "metal::atanh";
        case TOperator::EOpFma:
            return "metal::fma";
        case TOperator::EOpPow:
            return "metal::powr";  // GLSL's pow excludes negative x
        case TOperator::EOpExp:
            return "metal::exp";
        case TOperator::EOpExp2:
            return "metal::exp2";
        case TOperator::EOpLog:
            return "metal::log";
        case TOperator::EOpLog2:
            return "metal::log2";
        case TOperator::EOpSqrt:
            return "metal::sqrt";
        case TOperator::EOpFloor:
            return "metal::floor";
        case TOperator::EOpTrunc:
            return "metal::trunc";
        case TOperator::EOpCeil:
            return "metal::ceil";
        case TOperator::EOpFract:
            return "metal::fract";
        case TOperator::EOpMin:
            return "metal::min";
        case TOperator::EOpMax:
            return "metal::max";
        case TOperator::EOpRound:
            return "metal::round";
        case TOperator::EOpRoundEven:
            return "metal::rint";
        case TOperator::EOpClamp:
            return "metal::clamp";  // TODO fast vs precise namespace
        case TOperator::EOpLoopForwardProgress:
            return "ANGLE_loopForwardProgress";
        case TOperator::EOpSaturate:
            return "metal::saturate";  // TODO fast vs precise namespace
        case TOperator::EOpMix:
            if (argType2 && argType2->getBasicType() == EbtBool)
            {
                return "metal::select";
            }
            return "metal::mix";
        case TOperator::EOpStep:
            return "metal::step";
        case TOperator::EOpSmoothstep:
            return "metal::smoothstep";
        case TOperator::EOpModf:
            return "metal::modf";
        case TOperator::EOpIsnan:
            return "metal::isnan";
        case TOperator::EOpIsinf:
            return "metal::isinf";
        case TOperator::EOpLdexp:
            return "metal::ldexp";
        case TOperator::EOpFrexp:
            return "ANGLE_frexp";
        case TOperator::EOpInversesqrt:
            return "metal::rsqrt";
        case TOperator::EOpCross:
            return "metal::cross";
        case TOperator::EOpDFdx:
            return "metal::dfdx";
        case TOperator::EOpDFdy:
            return "metal::dfdy";
        case TOperator::EOpFwidth:
            return "metal::fwidth";
        case TOperator::EOpTranspose:
            return "metal::transpose";
        case TOperator::EOpDeterminant:
            return "metal::determinant";

        case TOperator::EOpInverse:
            return "ANGLE_inverse";

        case TOperator::EOpInterpolateAtCentroid:
            return "ANGLE_interpolateAtCentroid";
        case TOperator::EOpInterpolateAtSample:
            return "ANGLE_interpolateAtSample";
        case TOperator::EOpInterpolateAtOffset:
            return "ANGLE_interpolateAtOffset";
        case TOperator::EOpInterpolateAtCenter:
            return "ANGLE_interpolateAtCenter";

        case TOperator::EOpFloatBitsToInt:
        case TOperator::EOpFloatBitsToUint:
        case TOperator::EOpIntBitsToFloat:
        case TOperator::EOpUintBitsToFloat:
        {
#define RETURN_AS_TYPE_SCALAR()             \
    do                                      \
        switch (resultType.getBasicType())  \
        {                                   \
            case TBasicType::EbtInt:        \
                return "as_type<int>";      \
            case TBasicType::EbtUInt:       \
                return "as_type<uint32_t>"; \
            case TBasicType::EbtFloat:      \
                return "as_type<float>";    \
            default:                        \
                UNIMPLEMENTED();            \
                return "TOperator_TODO";    \
        }                                   \
    while (false)

#define RETURN_AS_TYPE(post)                     \
    do                                           \
        switch (resultType.getBasicType())       \
        {                                        \
            case TBasicType::EbtInt:             \
                return "as_type<int" post ">";   \
            case TBasicType::EbtUInt:            \
                return "as_type<uint" post ">";  \
            case TBasicType::EbtFloat:           \
                return "as_type<float" post ">"; \
            default:                             \
                UNIMPLEMENTED();                 \
                return "TOperator_TODO";         \
        }                                        \
    while (false)

            if (resultType.isScalar())
            {
                RETURN_AS_TYPE_SCALAR();
            }
            else if (resultType.isVector())
            {
                switch (resultType.getNominalSize())
                {
                    case 2:
                        RETURN_AS_TYPE("2");
                    case 3:
                        RETURN_AS_TYPE("3");
                    case 4:
                        RETURN_AS_TYPE("4");
                    default:
                        UNREACHABLE();
                        return nullptr;
                }
            }
            else
            {
                UNIMPLEMENTED();
                return "TOperator_TODO";
            }

#undef RETURN_AS_TYPE
#undef RETURN_AS_TYPE_SCALAR
        }

        case TOperator::EOpPackUnorm2x16:
            return "metal::pack_float_to_unorm2x16";
        case TOperator::EOpPackSnorm2x16:
            return "metal::pack_float_to_snorm2x16";

        case TOperator::EOpPackUnorm4x8:
            return "metal::pack_float_to_unorm4x8";
        case TOperator::EOpPackSnorm4x8:
            return "metal::pack_float_to_snorm4x8";

        case TOperator::EOpUnpackUnorm2x16:
            return "metal::unpack_unorm2x16_to_float";
        case TOperator::EOpUnpackSnorm2x16:
            return "metal::unpack_snorm2x16_to_float";

        case TOperator::EOpUnpackUnorm4x8:
            return "metal::unpack_unorm4x8_to_float";
        case TOperator::EOpUnpackSnorm4x8:
            return "metal::unpack_snorm4x8_to_float";

        case TOperator::EOpPackHalf2x16:
            return "ANGLE_pack_half_2x16";
        case TOperator::EOpUnpackHalf2x16:
            return "ANGLE_unpack_half_2x16";

        case TOperator::EOpNumSamples:
            return "metal::get_num_samples";
        case TOperator::EOpSamplePosition:
            return "metal::get_sample_position";

        case TOperator::EOpUaddCarry:
            return "ANGLE_uaddCarry";
        case TOperator::EOpUsubBorrow:
            return "ANGLE_usubBorrow";
        case TOperator::EOpUmulExtended:
            return "ANGLE_umulExtended";
        case TOperator::EOpImulExtended:
            return "ANGLE_imulExtended";
        case TOperator::EOpBitfieldExtract:
            return "metal::extract_bits";
        case TOperator::EOpBitfieldInsert:
            return "metal::insert_bits";
        case TOperator::EOpBitfieldReverse:
            return "metal::reverse_bits";
        case TOperator::EOpBitCount:
            return "ANGLE_bitCount";
        case TOperator::EOpFindLSB:
            return "ANGLE_findLSB";
        case TOperator::EOpFindMSB:
            return "ANGLE_findMSB";
        case TOperator::EOpBarrier:
            return "ANGLE_barrier";
        case TOperator::EOpMemoryBarrier:
        case TOperator::EOpMemoryBarrierAtomicCounter:
        case TOperator::EOpMemoryBarrierBuffer:
        case TOperator::EOpMemoryBarrierShared:
        case TOperator::EOpGroupMemoryBarrier:
            return "ANGLE_memoryBarrier";
        case TOperator::EOpAtomicAdd:
            return "metal::atomic_fetch_add_explicit";
        case TOperator::EOpAtomicMin:
            return "metal::atomic_fetch_min_explicit";
        case TOperator::EOpAtomicMax:
            return "metal::atomic_fetch_max_explicit";
        case TOperator::EOpAtomicAnd:
            return "metal::atomic_fetch_and_explicit";
        case TOperator::EOpAtomicOr:
            return "metal::atomic_fetch_or_explicit";
        case TOperator::EOpAtomicXor:
            return "metal::atomic_fetch_xor_explicit";
        case TOperator::EOpAtomicExchange:
            return "metal::atomic_exchange_explicit";
        case TOperator::EOpAtomicCompSwap:
            return "ANGLE_atomicCompSwap";
        case TOperator::EOpEmitVertex:
        case TOperator::EOpEndPrimitive:
            UNIMPLEMENTED();
            return "TOperator_TODO";

        case TOperator::EOpArrayLength:
            // SSBO runtime-array length is emitted directly by visitUnary().
            return nullptr;

        case TOperator::EOpNull:
        case TOperator::EOpConstruct:
        case TOperator::EOpCallFunctionInAST:
        case TOperator::EOpCallInternalRawFunction:
        case TOperator::EOpIndexDirect:
        case TOperator::EOpIndexIndirect:
        case TOperator::EOpIndexDirectStruct:
        case TOperator::EOpIndexDirectInterfaceBlock:
            UNREACHABLE();
            return nullptr;
        default:
            // Any other built-in function.
            return nullptr;
    }
}

static bool IsSymbolicOperator(TOperator op,
                               const TType &resultType,
                               const TType *argType0,
                               const TType *argType1)
{
    const char *operatorString = GetOperatorString(op, resultType, argType0, argType1, nullptr);
    if (operatorString == nullptr)
    {
        return false;
    }
    return !std::isalnum(operatorString[0]);
}

static TIntermBinary *AsSpecificBinaryNode(TIntermNode &node, TOperator op)
{
    TIntermBinary *binaryNode = node.getAsBinaryNode();
    if (binaryNode)
    {
        return binaryNode->getOp() == op ? binaryNode : nullptr;
    }
    return nullptr;
}

static bool Parenthesize(TIntermNode &node)
{
    if (node.getAsSymbolNode())
    {
        return false;
    }
    if (node.getAsConstantUnion())
    {
        return false;
    }
    if (node.getAsAggregate())
    {
        return false;
    }
    if (node.getAsSwizzleNode())
    {
        return false;
    }

    if (TIntermUnary *unaryNode = node.getAsUnaryNode())
    {
        // TODO: Use a precedence and associativity rules instead of this ad-hoc impl.
        const TType &resultType = unaryNode->getType();
        const TType &argType    = unaryNode->getOperand()->getType();
        return IsSymbolicOperator(unaryNode->getOp(), resultType, &argType, nullptr);
    }

    if (TIntermBinary *binaryNode = node.getAsBinaryNode())
    {
        // TODO: Use a precedence and associativity rules instead of this ad-hoc impl.
        const TOperator op = binaryNode->getOp();
        switch (op)
        {
            case TOperator::EOpIndexDirectStruct:
            case TOperator::EOpIndexDirectInterfaceBlock:
            case TOperator::EOpIndexDirect:
            case TOperator::EOpIndexIndirect:
                return Parenthesize(*binaryNode->getLeft());

            case TOperator::EOpAssign:
            case TOperator::EOpInitialize:
                return AsSpecificBinaryNode(*binaryNode->getRight(), TOperator::EOpComma);

            case TOperator::EOpAddAssign:
            case TOperator::EOpSubAssign:
            case TOperator::EOpMulAssign:
            case TOperator::EOpDivAssign:
            case TOperator::EOpIModAssign:
            case TOperator::EOpBitShiftLeftAssign:
            case TOperator::EOpBitShiftRightAssign:
            case TOperator::EOpVectorTimesScalarAssign:
                // A compound assignment binds less tightly than any operator that can enclose it,
                // both in its symbolic form `a += b` and when it is emulated and emitted as an
                // assignment `a = ANGLE_addInt(a, b)`. The latter form has an alphanumeric operator
                // string, so it cannot be detected with IsSymbolicOperator().
                return true;

            default:
            {
                const TType &resultType = binaryNode->getType();
                const TType &leftType   = binaryNode->getLeft()->getType();
                const TType &rightType  = binaryNode->getRight()->getType();
                return IsSymbolicOperator(binaryNode->getOp(), resultType, &leftType, &rightType);
            }
        }
    }

    return true;
}

void GenMetalTraverser::groupedTraverse(TIntermNode &node)
{
    const bool emitParens = Parenthesize(node);

    if (emitParens)
    {
        mOut << "(";
    }

    node.traverse(this);

    if (emitParens)
    {
        mOut << ")";
    }
}

void GenMetalTraverser::emitPostQualifier(const EmitVariableDeclarationConfig &evdConfig,
                                          const VarDecl &decl,
                                          const TQualifier qualifier)
{
    bool isInvariant = false;
    switch (qualifier)
    {
        case TQualifier::EvqPosition:
            isInvariant = decl.type().isInvariant();
            [[fallthrough]];
        case TQualifier::EvqFragCoord:
            mOut << " [[position]]";
            break;

        case TQualifier::EvqClipDistance:
            mOut << " [[clip_distance]] [" << decl.type().getOutermostArraySize() << "]";
            break;

        case TQualifier::EvqPointSize:
            mOut << " [[point_size]]";
            break;

        case TQualifier::EvqVertexID:
            if (evdConfig.isMainParameter)
            {
                mOut << " [[vertex_id]]";
            }
            break;

        case TQualifier::EvqPointCoord:
            if (evdConfig.isMainParameter)
            {
                mOut << " [[point_coord]]";
            }
            break;

        case TQualifier::EvqFrontFacing:
            if (evdConfig.isMainParameter)
            {
                mOut << " [[front_facing]]";
            }
            break;

        case TQualifier::EvqSampleID:
            if (evdConfig.isMainParameter)
            {
                mOut << " [[sample_id]]";
            }
            break;

        case TQualifier::EvqSampleMaskIn:
            if (evdConfig.isMainParameter)
            {
                mOut << " [[sample_mask]]";
            }
            break;

        default:
            break;
    }

    if (isInvariant)
    {
        mOut << " [[invariant]]";

        TranslatorMetalReflection *reflection = mtl::getTranslatorMetalReflection(&mCompiler);
        reflection->hasInvariance             = true;
    }
}

void GenMetalTraverser::emitLoopBody(TIntermBlock *bodyNode)
{
    bodyNode->traverse(this);
}

static void EmitName(Sink &out, const Name &name, char userSymbolPrefix)
{
#if defined(ANGLE_ENABLE_ASSERTS)
    DebugSink::EscapedSink escapedOut(out.escape());
#else
    TInfoSinkBase &escapedOut = out;
#endif
    name.emit(escapedOut, userSymbolPrefix);
}

void GenMetalTraverser::emitNameOf(const TField &object)
{
    EmitName(mOut, Name(object), kUserVariableNamePrefix);
}

void GenMetalTraverser::emitNameOf(const TSymbol &object)
{
    ASSERT(mRenamedSymbols.find(&object) == mRenamedSymbols.end());
    EmitName(mOut, Name(object), kUserVariableNamePrefix);
}

void GenMetalTraverser::emitNameOf(const TVariable &object)
{
    auto it = mRenamedSymbols.find(&object);
    if (it == mRenamedSymbols.end())
    {
        // If this is a private variable and we're not interested in retaining debug info, use a
        // temp name.
        if (!mCompileOptions.outputDebugInfo && object.symbolType() == SymbolType::UserDefined &&
            IsPrivate(object.getType().getQualifier()))
        {
            EmitName(mOut, Name(object.getTempName(), SymbolType::AngleInternal),
                     kUserVariableNamePrefix);
        }
        else
        {
            EmitName(mOut, Name(object), kUserVariableNamePrefix);
        }
    }
    else
    {
        EmitName(mOut, it->second, kUserVariableNamePrefix);
    }
}

void GenMetalTraverser::emitBlockNameOf(const TSymbol &object)
{
    ASSERT(mRenamedSymbols.find(&object) == mRenamedSymbols.end());
    EmitName(mOut, Name(object), kUserBlockNamePrefix);
}

void GenMetalTraverser::emitNameOf(const VarDecl &object)
{
    if (object.isField())
    {
        emitNameOf(object.field());
    }
    else
    {
        emitNameOf(object.variable());
    }
}

void GenMetalTraverser::emitBareTypeName(const TType &type, const EmitTypeConfig &etConfig)
{
    const TBasicType basicType = type.getBasicType();

    switch (basicType)
    {
        case TBasicType::EbtVoid:
        case TBasicType::EbtBool:
        case TBasicType::EbtFloat:
        case TBasicType::EbtInt:
        {
            mOut << type.getBasicString();
            break;
        }
        case TBasicType::EbtUInt:
        {
            if (type.isScalar())
            {
                mOut << "uint32_t";
            }
            else
            {
                mOut << type.getBasicString();
            }
        }
        break;

        case TBasicType::EbtStruct:
        {
            const TStructure &structure = *type.getStruct();
            if (structure.isImplementingInterfaceBlock())
            {
                emitBlockNameOf(structure);
            }
            else
            {
                emitNameOf(structure);
            }
        }
        break;

        case TBasicType::EbtInterfaceBlock:
        {
            const TInterfaceBlock &interfaceBlock = *type.getInterfaceBlock();
            emitBlockNameOf(interfaceBlock);
        }
        break;

        default:
        {
            if (IsSampler(basicType))
            {
                if (etConfig.evdConfig && etConfig.evdConfig->isMainParameter)
                {
                    EmitName(mOut, GetTextureTypeName(basicType), kUserVariableNamePrefix);
                }
                else
                {
                    const TStructure &env = mSymbolEnv.getTextureEnv(basicType);
                    emitNameOf(env);
                }
            }
            else if (IsImage(basicType))
            {
                const char *textureType;
                const char *componentType;
                switch (type.getBasicType())
                {
                    case EbtImageBuffer:
                        textureType   = "texture_buffer";
                        componentType = "float";
                        break;
                    case EbtIImageBuffer:
                        textureType   = "texture_buffer";
                        componentType = "int";
                        break;
                    case EbtUImageBuffer:
                        textureType   = "texture_buffer";
                        componentType = "uint";
                        break;
                    case EbtImage2D:
                        textureType   = "texture2d";
                        componentType = "float";
                        break;
                    case EbtIImage2D:
                        textureType   = "texture2d";
                        componentType = "int";
                        break;
                    case EbtUImage2D:
                        textureType   = "texture2d";
                        componentType = "uint";
                        break;
                    case EbtImage2DArray:
                        textureType   = "texture2d_array";
                        componentType = "float";
                        break;
                    case EbtIImage2DArray:
                        textureType   = "texture2d_array";
                        componentType = "int";
                        break;
                    case EbtUImage2DArray:
                        textureType   = "texture2d_array";
                        componentType = "uint";
                        break;
                    case EbtImage3D:
                        textureType   = "texture3d";
                        componentType = "float";
                        break;
                    case EbtIImage3D:
                        textureType   = "texture3d";
                        componentType = "int";
                        break;
                    case EbtUImage3D:
                        textureType   = "texture3d";
                        componentType = "uint";
                        break;
                    case EbtImageCube:
                        textureType   = mCompileOptions.emulateR32fImageAtomicExchange
                                            ? "texture2d_array"
                                            : "texturecube";
                        componentType = "float";
                        break;
                    case EbtIImageCube:
                        textureType   = mCompileOptions.emulateR32fImageAtomicExchange
                                            ? "texture2d_array"
                                            : "texturecube";
                        componentType = "int";
                        break;
                    case EbtUImageCube:
                        textureType   = mCompileOptions.emulateR32fImageAtomicExchange
                                            ? "texture2d_array"
                                            : "texturecube";
                        componentType = "uint";
                        break;
                    default:
                        UNIMPLEMENTED();
                        textureType   = "texture2d";
                        componentType = "float";
                        break;
                }

                const char *access = "read_write";
                if (type.getMemoryQualifier().readonly)
                {
                    access = "read";
                }
                else if (type.getMemoryQualifier().writeonly)
                {
                    access = "write";
                }

                const bool isCoherent = type.getMemoryQualifier().coherent;
                if (isCoherent)
                {
                    // Metal 3.2 defaults textures to threadgroup coherence.  GLSL coherent image
                    // accesses need visibility across the device when synchronized.
                    mOut << "/*ANGLE_REQUIRES_MSL_3_2*/";
                }
                mOut << "metal::" << textureType << "<" << componentType
                     << ", metal::access::" << access;
                if (isCoherent)
                {
                    mOut << ", metal::memory_coherence_device";
                }
                mOut << ">";
            }
            else
            {
                UNIMPLEMENTED();
            }
        }
    }
}

void GenMetalTraverser::emitType(const TType &type, const EmitTypeConfig &etConfig)
{
    const bool isUBO = etConfig.evdConfig ? etConfig.evdConfig->isUBO : false;
    const bool isRuntimeArrayPointer =
        etConfig.evdConfig && etConfig.evdConfig->isPointer && type.isUnsizedArray();
    TType typeToEmit = type;
    if (isRuntimeArrayPointer)
    {
        // Metal represents a GLSL runtime array as a pointer to its element type.  The array
        // itself has no compile-time size and must not be emitted as metal::array<T, 0>.
        typeToEmit.toArrayElementType();
    }
    if (etConfig.evdConfig)
    {
        const auto &evdConfig = *etConfig.evdConfig;
        if (isUBO)
        {
            if (type.isArray())
            {
                EmitMetalArrayTypePrefix(mOut, type.getArraySizes());
            }
        }
        if (evdConfig.isPointer)
        {
            mOut << toString(*evdConfig.isPointer);
            mOut << " ";
        }
        else if (evdConfig.isReference)
        {
            mOut << toString(*evdConfig.isReference);
            mOut << " ";
        }
        else if (evdConfig.storageAddressSpace)
        {
            mOut << toString(*evdConfig.storageAddressSpace);
            mOut << " ";
        }
    }

    if (!isUBO)
    {
        if (typeToEmit.isArray())
        {
            EmitMetalArrayTypePrefix(mOut, typeToEmit.getArraySizes());
        }
    }

    if (type.isInterpolant())
    {
        mOut << "metal::interpolant<";
    }

    if (type.isVector() || type.isMatrix())
    {
        mOut << "metal::";
    }

    if (etConfig.evdConfig && etConfig.evdConfig->isPacked)
    {
        mOut << "packed_";
    }

    emitBareTypeName(typeToEmit, etConfig);

    if (typeToEmit.isVector())
    {
        mOut << static_cast<uint32_t>(typeToEmit.getNominalSize());
    }
    else if (typeToEmit.isMatrix())
    {
        mOut << static_cast<uint32_t>(typeToEmit.getCols()) << "x"
             << static_cast<uint32_t>(typeToEmit.getRows());
    }

    if (typeToEmit.isInterpolant())
    {
        mOut << ", metal::interpolation::";
        switch (type.getQualifier())
        {
            case EvqNoPerspectiveIn:
            case EvqNoPerspectiveCentroidIn:
            case EvqNoPerspectiveSampleIn:
                mOut << "no_";
                break;
            default:
                break;
        }
        mOut << "perspective>";
    }

    if (!isUBO)
    {
        if (typeToEmit.isArray())
        {
            // Represent a runtime-sized SSBO member as a one-element trailing array. The array's
            // GLSL type remains unsized, so visitBinary emits unclamped indexing; this member
            // provides the correct base offset in the bound Metal buffer.
            EmitMetalArrayTypeSuffix(mOut, typeToEmit.getArraySizes(), type.isUnsizedArray());
        }
    }

    if (etConfig.evdConfig)
    {
        const auto &evdConfig = *etConfig.evdConfig;
        if (evdConfig.isPointer)
        {
            mOut << " *";
        }
        else if (evdConfig.isReference)
        {
            mOut << " &";
        }
        if (isUBO)
        {
            if (type.isArray())
            {
                EmitMetalArrayTypeSuffix(mOut, type.getArraySizes());
            }
        }
    }
}

void GenMetalTraverser::emitFieldDeclaration(const TField &field,
                                             const TStructure &parent,
                                             FieldAnnotationIndices &annotationIndices)
{
    const TType &type      = *field.type();
    const TBasicType basic = type.getBasicType();

    if (IsImage(basic) && type.isArray())
    {
        // MSL texture arrays can be function arguments, but they cannot be members of a regular
        // structure.  ANGLE passes image resources through a structure, so expand each GLSL image
        // array element into a separately bound texture member.  visitBinary maps constant GLSL
        // indexing to these generated members.
        const uint32_t imageArraySize = static_cast<uint32_t>(type.getArraySizeProduct());
        TType imageType               = type;
        imageType.toArrayElementType();

        TranslatorMetalReflection *reflection = mtl::getTranslatorMetalReflection(&mCompiler);
        for (uint32_t arrayElement = 0; arrayElement < imageArraySize; ++arrayElement)
        {
            if (arrayElement > 0)
            {
                mOut << ";\n";
                emitIndentation();
            }

            emitType(imageType, {});
            mOut << " ";
            emitNameOf(field);
            mOut << "_ANGLE_image_element_" << arrayElement;
            mOut << " [[";
            if (type.getLayoutQualifier().rasterOrdered)
            {
                mOut << "raster_order_group(0), ";
            }
            mOut << "texture(" << mMainTextureIndex + arrayElement << ")]]";
            reflection->addRWTextureBinding(field.name().data(), arrayElement,
                                            static_cast<int>(mMainTextureIndex + arrayElement));
            if (IsImage2DType(imageType.getBasicType()))
            {
                mOut << ";\n";
                emitIndentation();
                TType image3DType = imageType;
                image3DType.setBasicType(GetImage3DType(imageType.getBasicType()));
                emitType(image3DType, {});
                mOut << " ";
                emitNameOf(field);
                mOut << "_ANGLE_image_element_" << arrayElement << "_ANGLE_3D [[texture("
                     << mMainTextureIndex + arrayElement + kMetalImage3DBindingOffset << ")]]";
                reflection->addRWTexture3DBinding(
                    field.name().data(), arrayElement,
                    static_cast<int>(mMainTextureIndex + arrayElement +
                                     kMetalImage3DBindingOffset));
            }
        }
        mMainTextureIndex += imageArraySize;
        return;
    }

    if (IsImage2DType(type.getBasicType()))
    {
        TranslatorMetalReflection *reflection = mtl::getTranslatorMetalReflection(&mCompiler);
        emitType(type, {});
        mOut << " ";
        emitNameOf(field);
        mOut << " [[texture(" << mMainTextureIndex << ")]];\n";
        emitIndentation();

        TType image3DType = type;
        image3DType.setBasicType(GetImage3DType(type.getBasicType()));
        emitType(image3DType, {});
        mOut << " ";
        emitNameOf(field);
        mOut << "_ANGLE_3D [[texture(" << mMainTextureIndex + kMetalImage3DBindingOffset << ")]]";
        reflection->addRWTextureBinding(field.name().data(), 0,
                                        static_cast<int>(mMainTextureIndex));
        reflection->addRWTexture3DBinding(
            field.name().data(), 0,
            static_cast<int>(mMainTextureIndex + kMetalImage3DBindingOffset));
        ++mMainTextureIndex;
        return;
    }

    EmitVariableDeclarationConfig evdConfig;
    evdConfig.emitPostQualifier      = true;
    evdConfig.disableStructSpecifier = true;
    TType baseType(type);
    baseType.toArrayBaseType();
    const TStructure *nestedStorageBufferStruct = baseType.getStruct();
    const bool usesStorageBufferStructVariant =
        parent.isImplementingShaderStorageBlock() && nestedStorageBufferStruct &&
        NeedsStorageBufferStructVariant(*nestedStorageBufferStruct,
                                        Overlay(TLayoutBlockStorage::EbsStd430, type),
                                        Overlay(TLayoutMatrixPacking::EmpColumnMajor, type));
    const bool usesStorageBufferMatrixWrapper =
        parent.isImplementingShaderStorageBlock() && NeedsStorageBufferMatrixWrapper(type);
    if (usesStorageBufferStructVariant)
    {
        const TLayoutBlockStorage storage  = Overlay(TLayoutBlockStorage::EbsStd430, type);
        const TLayoutMatrixPacking packing = Overlay(TLayoutMatrixPacking::EmpColumnMajor, type);
        if (type.isArray())
        {
            EmitMetalArrayTypePrefix(mOut, type.getArraySizes());
        }
        mOut << getStorageBufferStructVariantName(*nestedStorageBufferStruct, storage, packing);
        if (type.isArray())
        {
            EmitMetalArrayTypeSuffix(mOut, type.getArraySizes(), type.isUnsizedArray());
        }
        mOut << " ";
        emitNameOf(field);
    }
    else if (usesStorageBufferMatrixWrapper)
    {
        if (type.isArray())
        {
            EmitMetalArrayTypePrefix(mOut, type.getArraySizes());
        }
        mOut << GetStorageBufferMatrixWrapperName(parent, field);
        if (type.isArray())
        {
            EmitMetalArrayTypeSuffix(mOut, type.getArraySizes(), type.isUnsizedArray());
        }
        mOut << " ";
        emitNameOf(field);
    }

    size_t storageBufferArrayStride = 0;
    const bool usesStorageBufferArrayWrapper =
        parent.isImplementingShaderStorageBlock() &&
        GetStorageBufferArrayWrapperStride(type, &storageBufferArrayStride);
    const bool isStorageBufferVec3 =
        parent.isImplementingShaderStorageBlock() &&
        (type.getLayoutQualifier().blockStorage == TLayoutBlockStorage::EbsStd430 ||
         type.getLayoutQualifier().blockStorage == TLayoutBlockStorage::EbsStd140 ||
         type.getLayoutQualifier().blockStorage == TLayoutBlockStorage::EbsShared) &&
        type.isVector() && type.getNominalSize() == 3 && !type.isArray() &&
        (type.getBasicType() == TBasicType::EbtFloat || type.getBasicType() == TBasicType::EbtInt ||
         type.getBasicType() == TBasicType::EbtUInt);
    if (isStorageBufferVec3)
    {
        // Metal's three-component vectors occupy 16 bytes.  std430 places a following scalar
        // after the vec3's 12-byte payload, so use an aligned packed vector in the block struct.
        mOut << "alignas(16) ";
    }
    evdConfig.isPacked = mSymbolEnv.isPacked(field) || isStorageBufferVec3;
    evdConfig.isUBO    = mSymbolEnv.isUBO(field);
    const bool isRuntimeStorageArray =
        parent.isImplementingShaderStorageBlock() && type.isUnsizedArray();
    evdConfig.isPointer   = isRuntimeStorageArray ? nullptr : mSymbolEnv.isPointer(field);
    evdConfig.isReference = mSymbolEnv.isReference(field);
    static constexpr AddressSpace kDeviceAddressSpace = AddressSpace::Device;
    if (!isRuntimeStorageArray && !evdConfig.isPointer && type.isUnsizedArray())
    {
        evdConfig.isPointer = &kDeviceAddressSpace;
    }
    if (usesStorageBufferStructVariant)
    {
        // Nested storage-block structures use a layout-specific padded MSL type.
    }
    else if (usesStorageBufferMatrixWrapper)
    {
        // Matrix fields use a structure with explicit GLSL row or column strides.
    }
    else if (usesStorageBufferArrayWrapper)
    {
        mOut << "metal::array<"
             << GetStorageBufferArrayWrapperName(parent, field,
                                                 Overlay(TLayoutBlockStorage::EbsStd430, type))
             << ", " << type.getOutermostArraySize() << "> ";
        emitNameOf(field);
    }
    else
    {
        emitVariableDeclaration(VarDecl(field), evdConfig);
    }

    const TQualifier qual                   = type.getQualifier();
    const TLayoutQualifier &layoutQualifier = type.getLayoutQualifier();
    const bool isExternalVarying            = mPipelineStructs.vertexOut.external == &parent ||
                                              mPipelineStructs.fragmentIn.external == &parent;
    const bool hasExplicitVaryingLocation = isExternalVarying && layoutQualifier.locationsSpecified;
    bool varyingLocationEmitted           = false;
    auto appendVaryingLocation            = [&]() {
        if (hasExplicitVaryingLocation)
        {
            mOut << ", user(locn" << layoutQualifier.location << ")";
            varyingLocationEmitted = true;
        }
    };
    switch (qual)
    {
        case TQualifier::EvqFlatIn:
            if (mPipelineStructs.fragmentIn.external == &parent)
            {
                mOut << " [[flat";
                appendVaryingLocation();
                mOut << "]]";
                TranslatorMetalReflection *reflection =
                    mtl::getTranslatorMetalReflection(&mCompiler);
                reflection->hasFlatInput = true;
            }
            break;

        case TQualifier::EvqNoPerspectiveIn:
            if (mPipelineStructs.fragmentIn.external == &parent && !type.isInterpolant())
            {
                mOut << " [[center_no_perspective";
                appendVaryingLocation();
                mOut << "]]";
            }
            break;

        case TQualifier::EvqCentroidIn:
            if (mPipelineStructs.fragmentIn.external == &parent && !type.isInterpolant())
            {
                mOut << " [[centroid_perspective";
                appendVaryingLocation();
                mOut << "]]";
            }
            break;

        case TQualifier::EvqSampleIn:
            if (mPipelineStructs.fragmentIn.external == &parent && !type.isInterpolant())
            {
                mOut << " [[sample_perspective";
                appendVaryingLocation();
                mOut << "]]";
            }
            break;

        case TQualifier::EvqNoPerspectiveCentroidIn:
            if (mPipelineStructs.fragmentIn.external == &parent && !type.isInterpolant())
            {
                mOut << " [[centroid_no_perspective";
                appendVaryingLocation();
                mOut << "]]";
            }
            break;

        case TQualifier::EvqNoPerspectiveSampleIn:
            if (mPipelineStructs.fragmentIn.external == &parent && !type.isInterpolant())
            {
                mOut << " [[sample_no_perspective";
                appendVaryingLocation();
                mOut << "]]";
            }
            break;

        case TQualifier::EvqFragColor:
            mOut << " [[color(0)]]";
            break;

        case TQualifier::EvqSecondaryFragColorEXT:
        case TQualifier::EvqSecondaryFragDataEXT:
            mOut << " [[color(0), index(1)]]";
            break;

        case TQualifier::EvqFragmentOut:
        case TQualifier::EvqFragmentInOut:
        case TQualifier::EvqFragData:
            if (mPipelineStructs.fragmentOut.external == &parent ||
                mPipelineStructs.fragmentOut.externalExtra == &parent)
            {
                if ((type.isVector() &&
                     (basic == TBasicType::EbtInt || basic == TBasicType::EbtUInt ||
                      basic == TBasicType::EbtFloat)) ||
                    qual == EvqFragData)
                {
                    // The OpenGL ES 3.0 spec says locations must be specified
                    // unless there is only a single output, in which case the
                    // location is 0. So, when we get to this point the shader
                    // will have been rejected if locations are not specified
                    // and there is more than one output.
                    if (layoutQualifier.locationsSpecified)
                    {
                        mOut << " [[color(" << layoutQualifier.location << ")";
                        ASSERT(layoutQualifier.index >= -1 && layoutQualifier.index <= 1);
                        if (layoutQualifier.index == 1)
                        {
                            mOut << ", index(1)";
                        }
                    }
                    else if (qual == EvqFragData)
                    {
                        mOut << " [[color(" << annotationIndices.color++ << ")";
                    }
                    else
                    {
                        // Either the only output or EXT_blend_func_extended is used;
                        // actual assignment will happen in UpdateFragmentShaderOutputs.
                        mOut << " [[" << sh::kUnassignedFragmentOutputString;
                    }
                    if (mRasterOrderGroupsSupported && qual == TQualifier::EvqFragmentInOut)
                    {
                        // Put fragment inouts in their own raster order group for better
                        // parallelism.
                        // NOTE: this is not required for the reads to be ordered and coherent.
                        // TODO(anglebug.com/40096838): Consider making raster order groups a PLS
                        // layout qualifier?
                        mOut << ", raster_order_group(0)";
                    }
                    mOut << "]]";
                }
            }
            break;

        case TQualifier::EvqFragDepth:
            mOut << " [[depth(";
            switch (type.getLayoutQualifier().depth)
            {
                case EdGreater:
                    mOut << "greater";
                    break;
                case EdLess:
                    mOut << "less";
                    break;
                default:
                    mOut << "any";
                    break;
            }
            mOut << "), function_constant(" << sh::mtl::kDepthWriteEnabledConstName << ")]]";
            break;

        case TQualifier::EvqSampleMask:
            if (field.symbolType() == SymbolType::AngleInternal)
            {
                if (mWriteHelperSampleMask)
                {
                    mOut << " [[sample_mask]]";
                }
                else
                {
                    mOut << " [[sample_mask, function_constant("
                         << sh::mtl::kSampleMaskWriteEnabledConstName << ")]]";
                }
            }
            break;

        default:
            break;
    }

    if (hasExplicitVaryingLocation && !varyingLocationEmitted)
    {
        mOut << " [[user(locn" << layoutQualifier.location << ")]]";
    }

    if (IsImage(type.getBasicType()))
    {
        const uint32_t imageArraySize =
            type.isArray() ? static_cast<uint32_t>(type.getArraySizeProduct()) : 1;
        TranslatorMetalReflection *reflection = mtl::getTranslatorMetalReflection(&mCompiler);
        for (uint32_t arrayElement = 0; arrayElement < imageArraySize; ++arrayElement)
        {
            mOut << " [[";
            if (type.getLayoutQualifier().rasterOrdered)
            {
                mOut << "raster_order_group(0), ";
            }
            mOut << "texture(" << mMainTextureIndex + arrayElement << ")]]";
            reflection->addRWTextureBinding(field.name().data(), arrayElement,
                                            static_cast<int>(mMainTextureIndex + arrayElement));
        }
        mMainTextureIndex += imageArraySize;
    }
}

static std::map<Name, size_t> BuildExternalAttributeIndexMap(
    const TCompiler &compiler,
    const PipelineScoped<TStructure> &structure)
{
    ASSERT(structure.isTotallyFull());

    const auto &shaderVars     = compiler.getAttributes();
    const size_t shaderVarSize = shaderVars.size();
    size_t shaderVarIndex      = 0;

    const auto &externalFields = structure.external->fields();
    const size_t externalSize  = externalFields.size();
    size_t externalIndex       = 0;

    const auto &internalFields = structure.internal->fields();
    const size_t internalSize  = internalFields.size();
    size_t internalIndex       = 0;

    // Internal fields are never split. External fields are sometimes split.
    ASSERT(externalSize >= internalSize);

    // Structures do not contain any inactive fields.
    ASSERT(shaderVarSize >= internalSize);

    std::map<Name, size_t> externalNameToAttributeIndex;
    size_t attributeIndex = 0;

    while (internalIndex < internalSize)
    {
        const TField &internalField = *internalFields[internalIndex];
        const Name internalName     = Name(internalField);
        const TType &internalType   = *internalField.type();
        while (internalName.rawName() != shaderVars[shaderVarIndex].name &&
               internalName.rawName() != shaderVars[shaderVarIndex].mappedName)
        {
            // This case represents an inactive field.

            ++shaderVarIndex;
            ASSERT(shaderVarIndex < shaderVarSize);

            ++attributeIndex;  // TODO: Might need to increment more if shader var type is a matrix.
        }

        const size_t cols =
            (internalType.isMatrix() && !externalFields[externalIndex]->type()->isMatrix())
                ? internalType.getCols()
                : 1;

        for (size_t c = 0; c < cols; ++c)
        {
            const TField &externalField = *externalFields[externalIndex];
            const Name externalName     = Name(externalField);

            externalNameToAttributeIndex[externalName] = attributeIndex;

            ++externalIndex;
            ++attributeIndex;
        }

        ++shaderVarIndex;
        ++internalIndex;
    }

    ASSERT(shaderVarIndex <= shaderVarSize);
    ASSERT(externalIndex <= externalSize);  // less than if padding was introduced
    ASSERT(internalIndex == internalSize);

    return externalNameToAttributeIndex;
}

void GenMetalTraverser::emitAttributeDeclaration(const TField &field,
                                                 FieldAnnotationIndices &annotationIndices)
{
    EmitVariableDeclarationConfig evdConfig;
    evdConfig.disableStructSpecifier = true;
    emitVariableDeclaration(VarDecl(field), evdConfig);
    mOut << sh::kUnassignedAttributeString;
}

void GenMetalTraverser::emitUniformBufferDeclaration(const TField &field,
                                                     FieldAnnotationIndices &annotationIndices)
{
    EmitVariableDeclarationConfig evdConfig;
    evdConfig.disableStructSpecifier = true;
    evdConfig.isUBO                  = mSymbolEnv.isUBO(field);
    evdConfig.isPointer              = mSymbolEnv.isPointer(field);
    emitVariableDeclaration(VarDecl(field), evdConfig);
    mOut << "[[id(" << annotationIndices.attribute << ")]]";

    const TType &type   = *field.type();
    const int arraySize = type.isArray() ? type.getArraySizeProduct() : 1;

    TranslatorMetalReflection *reflection = mtl::getTranslatorMetalReflection(&mCompiler);
    ASSERT(type.getBasicType() == TBasicType::EbtStruct);
    const TStructure *structure    = type.getStruct();
    const std::string originalName = reflection->getOriginalName(structure->uniqueId().get());
    reflection->addUniformBufferBinding(
        originalName,
        {.bindIndex = annotationIndices.attribute, .arraySize = static_cast<size_t>(arraySize)});

    annotationIndices.attribute += arraySize;
}

void GenMetalTraverser::emitStorageBufferDeclaration(const TField &field,
                                                     FieldAnnotationIndices &annotationIndices)
{
    const TType &type                = *field.type();
    const AddressSpace *addressSpace = mSymbolEnv.isPointer(field);
    if (type.isArray() && addressSpace)
    {
        // Each array element is a separate GL buffer binding. Keep the device pointer inside the
        // MSL array instead of making one device pointer to an array of block structs.
        EmitMetalArrayTypePrefix(mOut, type.getArraySizes());
        mOut << toString(*addressSpace) << " ";
        TType baseType = type;
        baseType.toArrayBaseType();
        emitBareTypeName(baseType, EmitTypeConfig{nullptr});
        mOut << " *";
        EmitMetalArrayTypeSuffix(mOut, type.getArraySizes());
        mOut << " ";
        emitNameOf(field);
    }
    else
    {
        EmitVariableDeclarationConfig evdConfig;
        evdConfig.disableStructSpecifier = true;
        evdConfig.isPointer              = addressSpace;
        emitVariableDeclaration(VarDecl(field), evdConfig);
    }
    mOut << "[[id(" << annotationIndices.attribute << ")]]";

    const int arraySize = type.isArray() ? type.getArraySizeProduct() : 1;

    TranslatorMetalReflection *reflection = mtl::getTranslatorMetalReflection(&mCompiler);
    ASSERT(type.getBasicType() == TBasicType::EbtStruct);
    const TStructure *structure    = type.getStruct();
    const std::string originalName = structure->name().data();
    reflection->addStorageBufferBinding(
        originalName,
        {.bindIndex = annotationIndices.attribute, .arraySize = static_cast<size_t>(arraySize)});

    annotationIndices.attribute += arraySize;
}

void GenMetalTraverser::emitStorageBufferLengthDeclaration(
    const TField &field,
    FieldAnnotationIndices &annotationIndices)
{
    const TType &type   = *field.type();
    const int arraySize = type.isArray() ? type.getArraySizeProduct() : 1;

    if (type.isArray())
    {
        EmitMetalArrayTypePrefix(mOut, type.getArraySizes());
        mOut << "constant uint *";
        EmitMetalArrayTypeSuffix(mOut, type.getArraySizes());
        mOut << " ";
    }
    else
    {
        mOut << "constant uint *";
    }
    emitNameOf(field);
    mOut << "_length [[id(" << annotationIndices.attribute << ")]]";
    annotationIndices.attribute += arraySize;
}

void GenMetalTraverser::emitStorageBufferArrayWrapper(const TStructure &parent,
                                                      const TField &field,
                                                      TLayoutBlockStorage storage)
{
    size_t stride = 0;
    if (!GetStorageBufferArrayWrapperStride(*field.type(), &stride, storage))
    {
        return;
    }

    const std::string wrapperName = GetStorageBufferArrayWrapperName(parent, field, storage);
    if (!mEmittedStorageBufferArrayWrappers.insert(wrapperName).second)
    {
        return;
    }

    TType elementType(*field.type());
    elementType.toArrayElementType();
    mOut << "struct alignas(" << stride << ") " << wrapperName << " { ";
    emitType(elementType, {});
    mOut << " value; };\n";
}

std::string GenMetalTraverser::getStorageBufferStructVariantName(const TStructure &structure,
                                                                 TLayoutBlockStorage storage,
                                                                 TLayoutMatrixPacking packing) const
{
    return "ANGLE_SSBO_struct_" + std::to_string(structure.uniqueId().get()) + "_" +
           std::to_string(static_cast<int>(storage)) + "_" +
           std::to_string(static_cast<int>(packing));
}

void GenMetalTraverser::emitStorageBufferPadding(size_t byteCount, size_t padIndex)
{
    ASSERT(byteCount % sizeof(uint32_t) == 0);
    if (byteCount == 0)
    {
        return;
    }
    mOut << "  metal::array<uint, " << byteCount / sizeof(uint32_t) << "> _ANGLE_SSBO_pad_"
         << padIndex << ";\n";
}

void GenMetalTraverser::emitStorageBufferStructVariant(const TStructure &structure,
                                                       TLayoutBlockStorage storage,
                                                       TLayoutMatrixPacking packing)
{
    const std::string variantName = getStorageBufferStructVariantName(structure, storage, packing);
    if (!mEmittedStorageBufferStructVariants.insert(variantName).second)
    {
        return;
    }

    for (const TField *field : structure.fields())
    {
        const TLayoutBlockStorage fieldStorage  = Overlay(storage, *field->type());
        const TLayoutMatrixPacking fieldPacking = Overlay(packing, *field->type());
        TType baseType(*field->type());
        baseType.toArrayBaseType();
        if (const TStructure *nested = baseType.getStruct();
            nested && NeedsStorageBufferStructVariant(*nested, fieldStorage, fieldPacking))
        {
            emitStorageBufferStructVariant(*nested, fieldStorage, fieldPacking);
        }
        emitStorageBufferArrayWrapper(structure, *field, fieldStorage);
    }

    TType structureType(&structure, false);
    const Layout structureLayout = GetGlslStorageBufferLayoutOf(structureType, storage, packing);
    mOut << "struct alignas(" << structureLayout.alignOf << ") " << variantName << "\n{\n";

    size_t offset   = 0;
    size_t padIndex = 0;
    for (const TField *field : structure.fields())
    {
        const TType &fieldType                  = *field->type();
        const TLayoutBlockStorage fieldStorage  = Overlay(storage, fieldType);
        const TLayoutMatrixPacking fieldPacking = Overlay(packing, fieldType);
        const Layout fieldLayout =
            GetGlslStorageBufferLayoutOf(fieldType, fieldStorage, fieldPacking);
        const size_t fieldOffset = RoundUpToStorageAlignment(offset, fieldLayout.alignOf);
        emitStorageBufferPadding(fieldOffset - offset, padIndex++);
        mOut << "  ";

        TType baseType(fieldType);
        baseType.toArrayBaseType();
        const TStructure *nested = baseType.getStruct();
        if (nested && NeedsStorageBufferStructVariant(*nested, fieldStorage, fieldPacking))
        {
            if (fieldType.isArray())
            {
                EmitMetalArrayTypePrefix(mOut, fieldType.getArraySizes());
            }
            mOut << getStorageBufferStructVariantName(*nested, fieldStorage, fieldPacking);
            if (fieldType.isArray())
            {
                EmitMetalArrayTypeSuffix(mOut, fieldType.getArraySizes(),
                                         fieldType.isUnsizedArray());
            }
            mOut << " ";
            emitNameOf(*field);
        }
        else if (nested)
        {
            EmitVariableDeclarationConfig fieldConfig;
            fieldConfig.disableStructSpecifier = true;
            emitType(fieldType, EmitTypeConfig{&fieldConfig});
            mOut << " ";
            emitNameOf(*field);
        }
        else
        {
            size_t arrayStride = 0;
            if (GetStorageBufferArrayWrapperStride(fieldType, &arrayStride, fieldStorage))
            {
                mOut << "metal::array<"
                     << GetStorageBufferArrayWrapperName(structure, *field, fieldStorage) << ", "
                     << fieldType.getOutermostArraySize() << "> ";
                emitNameOf(*field);
            }
            else
            {
                EmitVariableDeclarationConfig fieldConfig;
                fieldConfig.disableStructSpecifier = true;
                const bool isVec3    = !fieldType.isArray() && fieldType.isVector() &&
                                       fieldType.getNominalSize() == 3 &&
                                       (fieldType.getBasicType() == TBasicType::EbtFloat ||
                                        fieldType.getBasicType() == TBasicType::EbtInt ||
                                        fieldType.getBasicType() == TBasicType::EbtUInt);
                fieldConfig.isPacked = isVec3;
                if (isVec3)
                {
                    mOut << "alignas(" << fieldLayout.alignOf << ") ";
                }
                emitType(fieldType, EmitTypeConfig{&fieldConfig});
                mOut << " ";
                emitNameOf(*field);
            }
        }
        mOut << ";\n";
        offset = fieldOffset + fieldLayout.sizeOf;
    }

    emitStorageBufferPadding(structureLayout.sizeOf - offset, padIndex);
    mOut << "};\n";
}

void GenMetalTraverser::emitStructDeclaration(const TType &type)
{
    ASSERT(type.getBasicType() == TBasicType::EbtStruct);
    ASSERT(type.isStructSpecifier());

    const TStructure &structure = *type.getStruct();
    if (structure.isImplementingShaderStorageBlock())
    {
        for (const TField *field : structure.fields())
        {
            const TLayoutBlockStorage storage =
                Overlay(TLayoutBlockStorage::EbsStd430, *field->type());
            const TLayoutMatrixPacking packing =
                Overlay(TLayoutMatrixPacking::EmpColumnMajor, *field->type());
            TType baseType(*field->type());
            baseType.toArrayBaseType();
            if (const TStructure *nested = baseType.getStruct();
                nested && NeedsStorageBufferStructVariant(*nested, storage, packing))
            {
                emitStorageBufferStructVariant(*nested, storage, packing);
            }
            emitStorageBufferArrayWrapper(structure, *field, storage);

            if (NeedsStorageBufferMatrixWrapper(*field->type()))
            {
                TType matrixType(*field->type());
                matrixType.toArrayBaseType();
                size_t majorCount;
                size_t vectorSize;
                size_t vectorStride;
                bool isRowMajor;
                GetStorageBufferMatrixLayout(matrixType, &majorCount, &vectorSize, &vectorStride,
                                             &isRowMajor);

                mOut << "struct alignas(" << vectorStride << ") "
                     << GetStorageBufferMatrixWrapperName(structure, *field) << " {\n";
                for (size_t major = 0; major < majorCount; ++major)
                {
                    TType vectorType(TBasicType::EbtFloat, static_cast<uint8_t>(vectorSize));
                    EmitVariableDeclarationConfig packedConfig;
                    packedConfig.isPacked = true;
                    mOut << "  alignas(" << vectorStride << ") ";
                    emitType(vectorType, EmitTypeConfig{&packedConfig});
                    mOut << " " << (isRowMajor ? "row" : "column") << major << ";\n";
                }
                mOut << "};\n";
                mStorageBufferMatrixFields.emplace_back(&structure, field);
            }
        }
    }

    mOut << "struct ";
    emitBareTypeName(type, {});

    mOut << "\n";
    emitOpenBrace();

    std::map<Name, size_t> fieldToAttributeIndex;
    const bool hasAttributeIndices      = mPipelineStructs.vertexIn.external == &structure;
    const bool hasUniformBufferIndicies = mPipelineStructs.uniformBuffers.external == &structure;
    const bool hasStorageBufferIndices  = mPipelineStructs.storageBuffers.external == &structure;
    const bool reclaimUnusedAttributeIndices = mCompiler.getShaderVersion() < 300;

    if (hasAttributeIndices)
    {
        // When attribute aliasing is supported, external attribute struct is filled post-link.
        if (mCompiler.supportsAttributeAliasing())
        {
            mtl::getTranslatorMetalReflection(&mCompiler)->hasAttributeAliasing = true;
            mOut << "@@Attrib-Bindings@@\n";
            emitCloseBrace();
            return;
        }

        fieldToAttributeIndex =
            BuildExternalAttributeIndexMap(mCompiler, mPipelineStructs.vertexIn);
    }

    FieldAnnotationIndices annotationIndices;
    size_t storageBufferOffset   = 0;
    size_t storageBufferPadIndex = 0;

    for (const TField *field : structure.fields())
    {
        size_t storageBufferFieldOffset = 0;
        Layout storageBufferFieldLayout = Layout::Identity();
        if (structure.isImplementingShaderStorageBlock())
        {
            storageBufferFieldLayout =
                GetGlslStorageBufferLayoutOf(*field->type(), TLayoutBlockStorage::EbsStd430,
                                             TLayoutMatrixPacking::EmpColumnMajor);
            storageBufferFieldOffset =
                RoundUpToStorageAlignment(storageBufferOffset, storageBufferFieldLayout.alignOf);
            emitStorageBufferPadding(storageBufferFieldOffset - storageBufferOffset,
                                     storageBufferPadIndex++);
        }

        emitIndentation();
        if (hasAttributeIndices)
        {
            const auto it = fieldToAttributeIndex.find(Name(*field));
            if (it == fieldToAttributeIndex.end())
            {
                ASSERT(field->symbolType() == SymbolType::AngleInternal);
                ASSERT(field->name().beginsWith("_"));
                ASSERT(angle::EndsWith(field->name().data(), "_pad"));
                emitFieldDeclaration(*field, structure, annotationIndices);
            }
            else
            {
                ASSERT(field->symbolType() != SymbolType::AngleInternal ||
                       !field->name().beginsWith("_") ||
                       !angle::EndsWith(field->name().data(), "_pad"));
                if (!reclaimUnusedAttributeIndices)
                {
                    annotationIndices.attribute = it->second;
                }
                emitAttributeDeclaration(*field, annotationIndices);
            }
        }
        else if (hasUniformBufferIndicies)
        {
            emitUniformBufferDeclaration(*field, annotationIndices);
        }
        else if (hasStorageBufferIndices)
        {
            emitStorageBufferDeclaration(*field, annotationIndices);
        }
        else
        {
            emitFieldDeclaration(*field, structure, annotationIndices);
        }
        mOut << ";\n";
        if (structure.isImplementingShaderStorageBlock())
        {
            storageBufferOffset = storageBufferFieldOffset + storageBufferFieldLayout.sizeOf;
        }
        if (hasStorageBufferIndices)
        {
            emitIndentation();
            emitStorageBufferLengthDeclaration(*field, annotationIndices);
            mOut << ";\n";
        }
    }

    emitCloseBrace();
}

void GenMetalTraverser::emitOrdinaryVariableDeclaration(
    const VarDecl &decl,
    const EmitVariableDeclarationConfig &evdConfig)
{
    EmitTypeConfig etConfig;
    etConfig.evdConfig = &evdConfig;

    const TType &type = decl.type();
    if (type.getQualifier() == TQualifier::EvqClipDistance)
    {
        // Clip distance output uses float[n] type instead of metal::array.
        // The element count is emitted after the post qualifier.
        ASSERT(type.getBasicType() == TBasicType::EbtFloat);
        mOut << "float";
    }
    else if (type.getQualifier() == TQualifier::EvqSampleID && evdConfig.isMainParameter)
    {
        // Metal's [[sample_id]] must be unsigned
        ASSERT(type.getBasicType() == TBasicType::EbtInt);
        mOut << "uint32_t";
    }
    else
    {
        emitType(type, etConfig);
    }
    if (decl.symbolType() != SymbolType::Empty)
    {
        mOut << " ";
        emitNameOf(decl);
    }
}

void GenMetalTraverser::emitVariableDeclaration(const VarDecl &decl,
                                                const EmitVariableDeclarationConfig &evdConfig)
{
    const SymbolType symbolType = decl.symbolType();
    const TType &type           = decl.type();
    const TBasicType basicType  = type.getBasicType();

    switch (basicType)
    {
        case TBasicType::EbtStruct:
        {
            if (type.isStructSpecifier() && !evdConfig.disableStructSpecifier)
            {
                // It's invalid to declare a struct inside a function argument. When emitting a
                // function parameter, the callsite should set evdConfig.disableStructSpecifier.
                ASSERT(!evdConfig.isParameter);
                emitStructDeclaration(type);
                if (symbolType != SymbolType::Empty)
                {
                    mOut << " ";
                    emitNameOf(decl);
                }
            }
            else
            {
                emitOrdinaryVariableDeclaration(decl, evdConfig);
            }
        }
        break;

        default:
        {
            ASSERT(symbolType != SymbolType::Empty || evdConfig.isParameter);
            emitOrdinaryVariableDeclaration(decl, evdConfig);
        }
    }

    if (evdConfig.emitPostQualifier)
    {
        emitPostQualifier(evdConfig, decl, type.getQualifier());
    }
}

void GenMetalTraverser::visitSymbol(TIntermSymbol *symbolNode)
{
    const TVariable &var = symbolNode->variable();
    const TType &type    = var.getType();
    ASSERT(var.symbolType() != SymbolType::Empty);

    if (mCompiler.getShaderType() == GL_COMPUTE_SHADER)
    {
        switch (type.getQualifier())
        {
            case TQualifier::EvqGlobalInvocationID:
                mOut << "ANGLE_globalInvocationID";
                return;
            case TQualifier::EvqLocalInvocationID:
                mOut << "ANGLE_localInvocationID";
                return;
            case TQualifier::EvqLocalInvocationIndex:
                mOut << "ANGLE_localInvocationIndex";
                return;
            case TQualifier::EvqNumWorkGroups:
                mOut << "ANGLE_numWorkGroups";
                return;
            case TQualifier::EvqWorkGroupID:
                mOut << "ANGLE_workGroupID";
                return;
            case TQualifier::EvqWorkGroupSize:
            {
                const sh::WorkGroupSize &localSize = mCompiler.getComputeShaderLocalSize();
                mOut << "metal::uint3(" << localSize[0] << ", " << localSize[1] << ", "
                     << localSize[2] << ")";
                return;
            }
            default:
                break;
        }

        if (type.getQualifier() == TQualifier::EvqShared)
        {
            if (mIsComputeMain)
            {
                emitNameOf(var);
            }
            else
            {
                mOut << "(*";
                emitNameOf(var);
                mOut << ")";
            }
            return;
        }
    }

    if (type.getBasicType() == TBasicType::EbtVoid)
    {
        mOut << "/*";
        emitNameOf(var);
        mOut << "*/";
    }
    else
    {
        emitNameOf(var);
    }
}

void GenMetalTraverser::emitSingleConstant(const TConstantUnion *const constUnion)
{
    switch (constUnion->getType())
    {
        case TBasicType::EbtBool:
        {
            mOut << (constUnion->getBConst() ? "true" : "false");
        }
        break;

        case TBasicType::EbtFloat:
        {
            float value = constUnion->getFConst();
            if (std::isnan(value))
            {
                mOut << "NAN";
            }
            else if (std::isinf(value))
            {
                if (value < 0)
                {
                    mOut << "-";
                }
                mOut << "INFINITY";
            }
            else
            {
                mOut << value << "f";
            }
        }
        break;

        case TBasicType::EbtInt:
        {
            mOut << constUnion->getIConst();
        }
        break;

        case TBasicType::EbtUInt:
        {
            mOut << constUnion->getUConst() << "u";
        }
        break;

        default:
        {
            UNIMPLEMENTED();
        }
    }
}

const TConstantUnion *GenMetalTraverser::emitConstantUnionArray(
    const TConstantUnion *const constUnion,
    const size_t size)
{
    const TConstantUnion *constUnionIterated = constUnion;
    for (size_t i = 0; i < size; i++, ANGLE_UNSAFE_TODO(constUnionIterated++))
    {
        emitSingleConstant(constUnionIterated);

        if (i != size - 1)
        {
            mOut << ", ";
        }
    }
    return constUnionIterated;
}

const TConstantUnion *GenMetalTraverser::emitConstantUnion(const TType &type,
                                                           const TConstantUnion *constUnionBegin)
{
    const TConstantUnion *constUnionCurr = constUnionBegin;
    const TStructure *structure          = type.getStruct();
    if (structure)
    {
        EmitTypeConfig config = EmitTypeConfig{nullptr};
        emitType(type, config);
        mOut << "{";
        const TFieldList &fields = structure->fields();
        for (size_t i = 0; i < fields.size(); ++i)
        {
            const TType *fieldType = fields[i]->type();
            constUnionCurr         = emitConstantUnion(*fieldType, constUnionCurr);
            if (i != fields.size() - 1)
            {
                mOut << ", ";
            }
        }
        mOut << "}";
    }
    else
    {
        size_t size    = type.getObjectSize();
        bool writeType = size > 1;
        if (writeType)
        {
            EmitTypeConfig config = EmitTypeConfig{nullptr};
            emitType(type, config);
            mOut << "(";
        }
        constUnionCurr = emitConstantUnionArray(constUnionCurr, size);
        if (writeType)
        {
            mOut << ")";
        }
    }
    return constUnionCurr;
}

void GenMetalTraverser::visitConstantUnion(TIntermConstantUnion *constValueNode)
{
    emitConstantUnion(constValueNode->getType(), constValueNode->getConstantValue());
}

bool GenMetalTraverser::visitSwizzle(Visit, TIntermSwizzle *swizzleNode)
{
    groupedTraverse(*swizzleNode->getOperand());
    mOut << ".";

    {
#if defined(ANGLE_ENABLE_ASSERTS)
        DebugSink::EscapedSink escapedOut(mOut.escape());
        TInfoSinkBase &out = escapedOut.get();
#else
        TInfoSinkBase &out = mOut;
#endif
        swizzleNode->writeOffsetsAsXYZW(&out);
    }

    return false;
}

const TField &GenMetalTraverser::getDirectField(const TFieldListCollection &fieldListCollection,
                                                const TConstantUnion &index)
{
    ASSERT(index.getType() == TBasicType::EbtInt);

    const TFieldList &fieldList = fieldListCollection.fields();
    const int indexVal          = index.getIConst();
    const TField &field         = *fieldList[indexVal];

    return field;
}

const TField &GenMetalTraverser::getDirectField(const TIntermTyped &fieldsNode,
                                                TIntermTyped &indexNode)
{
    const TType &fieldsType = fieldsNode.getType();

    const TFieldListCollection *fieldListCollection = fieldsType.getStruct();
    if (fieldListCollection == nullptr)
    {
        fieldListCollection = fieldsType.getInterfaceBlock();
    }
    ASSERT(fieldListCollection);

    const TIntermConstantUnion *indexNode_ = indexNode.getAsConstantUnion();
    ASSERT(indexNode_);
    const TConstantUnion &index = *indexNode_->getConstantValue();

    return getDirectField(*fieldListCollection, index);
}

const TField *GenMetalTraverser::getStorageBufferField(TIntermTyped &expression,
                                                       const TStructure **parentOut)
{
    TIntermTyped *cursor = &expression;
    while (TIntermBinary *binary = cursor->getAsBinaryNode())
    {
        if (binary->getOp() == TOperator::EOpIndexDirectStruct ||
            binary->getOp() == TOperator::EOpIndexDirectInterfaceBlock)
        {
            const TStructure *structure = binary->getLeft()->getType().getStruct();
            if (structure && structure->isImplementingShaderStorageBlock())
            {
                if (parentOut)
                {
                    *parentOut = structure;
                }
                return &getDirectField(*binary->getLeft(), *binary->getRight());
            }
        }
        cursor = binary->getLeft();
    }
    return nullptr;
}

const TField *GenMetalTraverser::getStorageBufferArrayField(TIntermTyped &expression)
{
    const TField *arrayField       = nullptr;
    TIntermTyped *cursor           = &expression;
    TLayoutBlockStorage storage    = TLayoutBlockStorage::EbsStd430;
    bool isStorageBufferExpression = false;

    while (TIntermBinary *binary = cursor->getAsBinaryNode())
    {
        if (binary->getOp() == TOperator::EOpIndexDirectStruct ||
            binary->getOp() == TOperator::EOpIndexDirectInterfaceBlock)
        {
            const TStructure *structure = binary->getLeft()->getType().getStruct();
            if (structure)
            {
                const TField &field = getDirectField(*binary->getLeft(), *binary->getRight());
                if (!arrayField && field.type()->isArray())
                {
                    arrayField = &field;
                }
                if (structure->isImplementingShaderStorageBlock())
                {
                    storage = Overlay(TLayoutBlockStorage::EbsStd430, *field.type());
                    isStorageBufferExpression = true;
                }
            }
        }
        cursor = binary->getLeft();
    }

    size_t stride = 0;
    if (!isStorageBufferExpression || !arrayField ||
        !GetStorageBufferArrayWrapperStride(*arrayField->type(), &stride, storage))
    {
        return nullptr;
    }

    return arrayField;
}

bool GenMetalTraverser::visitBinary(Visit, TIntermBinary *binaryNode)
{
    const TOperator op      = binaryNode->getOp();
    TIntermTyped &leftNode  = *binaryNode->getLeft();
    TIntermTyped &rightNode = *binaryNode->getRight();

    const TStructure *matrixParent = nullptr;
    const TField *matrixField =
        leftNode.getType().isMatrix() ? getStorageBufferField(leftNode, &matrixParent) : nullptr;
    if (op == TOperator::EOpAssign && matrixField &&
        NeedsStorageBufferMatrixWrapper(*matrixField->type()))
    {
        mOut << GetStorageBufferMatrixWrapperName(*matrixParent, *matrixField) << "_store(&(";
        mSuppressStorageBufferMatrixLoad = true;
        groupedTraverse(leftNode);
        mSuppressStorageBufferMatrixLoad = false;
        mOut << "), ";
        groupedTraverse(rightNode);
        mOut << ")";
        return false;
    }

    const bool isMatrixStorageAccess =
        op == TOperator::EOpIndexDirectStruct || op == TOperator::EOpIndexDirectInterfaceBlock ||
        op == TOperator::EOpIndexDirect || op == TOperator::EOpIndexIndirect;
    if (!mSuppressStorageBufferMatrixLoad && isMatrixStorageAccess &&
        binaryNode->getType().isMatrix())
    {
        const TStructure *parent = nullptr;
        const TField *field      = getStorageBufferField(*binaryNode, &parent);
        if (field && NeedsStorageBufferMatrixWrapper(*field->type()))
        {
            mOut << GetStorageBufferMatrixWrapperName(*parent, *field) << "_load(";
            mSuppressStorageBufferMatrixLoad = true;
            groupedTraverse(*binaryNode);
            mSuppressStorageBufferMatrixLoad = false;
            mOut << ")";
            return false;
        }
    }

    // Metal's matrix *= overloads don't accept device-address-space matrices from SSBOs. Emit
    // matrix compound multiplication as a normal load, multiply, and store instead.
    if ((op == TOperator::EOpMatrixTimesMatrixAssign ||
         op == TOperator::EOpMatrixTimesScalarAssign) &&
        leftNode.getType().isMatrix())
    {
        if (matrixField && NeedsStorageBufferMatrixWrapper(*matrixField->type()))
        {
            const std::string wrapperName =
                GetStorageBufferMatrixWrapperName(*matrixParent, *matrixField);
            mOut << wrapperName << "_store(&(";
            mSuppressStorageBufferMatrixLoad = true;
            groupedTraverse(leftNode);
            mSuppressStorageBufferMatrixLoad = false;
            mOut << "), " << wrapperName << "_load(";
            mSuppressStorageBufferMatrixLoad = true;
            groupedTraverse(leftNode);
            mSuppressStorageBufferMatrixLoad = false;
            mOut << ") * ";
            groupedTraverse(rightNode);
            mOut << ")";
            return false;
        }

        groupedTraverse(leftNode);
        mOut << " = ";
        groupedTraverse(leftNode);
        mOut << " * ";
        groupedTraverse(rightNode);
        return false;
    }

    // Metal can lower dynamic vector-component assignments to a vector read/modify/write. Address
    // the scalar component directly so concurrent invocations updating different components do
    // not overwrite each other's updates.
    if (op == TOperator::EOpAssign)
    {
        TIntermBinary *componentAccess = leftNode.getAsBinaryNode();
        TIntermTyped *vectorExpression = componentAccess ? componentAccess->getLeft() : nullptr;
        if (componentAccess &&
            (componentAccess->getOp() == TOperator::EOpIndexDirect ||
             componentAccess->getOp() == TOperator::EOpIndexIndirect) &&
            vectorExpression && vectorExpression->getType().isVector())
        {
            const AddressSpace *vectorAddressSpace = nullptr;
            TIntermNode *vectorNode                = vectorExpression;
            while (vectorNode)
            {
                if (TIntermSymbol *symbol = vectorNode->getAsSymbolNode())
                {
                    if (symbol->getType().getQualifier() == TQualifier::EvqShared)
                    {
                        vectorAddressSpace = &kThreadgroupAddressSpace;
                    }
                    else
                    {
                        vectorAddressSpace = mSymbolEnv.isPointer(symbol->variable());
                    }
                    break;
                }

                TIntermBinary *fieldAccess = vectorNode->getAsBinaryNode();
                if (!fieldAccess)
                {
                    break;
                }
                if (fieldAccess->getOp() == TOperator::EOpIndexDirectStruct ||
                    fieldAccess->getOp() == TOperator::EOpIndexDirectInterfaceBlock)
                {
                    const TField &field =
                        getDirectField(*fieldAccess->getLeft(), *fieldAccess->getRight());
                    vectorAddressSpace = mSymbolEnv.isPointer(field);
                    if (vectorAddressSpace)
                    {
                        break;
                    }
                }
                vectorNode = fieldAccess->getLeft();
            }

            TIntermBinary *matrixColumn = vectorExpression->getAsBinaryNode();
            const bool isDeviceMatrixColumn =
                vectorAddressSpace && *vectorAddressSpace == AddressSpace::Device && matrixColumn &&
                (matrixColumn->getOp() == TOperator::EOpIndexDirect ||
                 matrixColumn->getOp() == TOperator::EOpIndexIndirect) &&
                matrixColumn->getLeft()->getType().isMatrix();
            if (isDeviceMatrixColumn ||
                (vectorAddressSpace && *vectorAddressSpace == AddressSpace::Threadgroup))
            {
                mOut << "reinterpret_cast<" << toString(*vectorAddressSpace) << " ";
                emitType(leftNode.getType(), {});
                mOut << " *>(&";
                groupedTraverse(*vectorExpression);
                mOut << ")[ANGLE_int_clamp(";
                groupedTraverse(*componentAccess->getRight());
                mOut << ", 0, "
                     << static_cast<uint32_t>(vectorExpression->getType().getNominalSize() - 1)
                     << ")] = ";
                groupedTraverse(rightNode);
                return false;
            }
        }
    }

    switch (op)
    {
        case TOperator::EOpIndexDirectStruct:
        case TOperator::EOpIndexDirectInterfaceBlock:
        {
            const TField &field = getDirectField(leftNode, rightNode);
            if (mSymbolEnv.isPointer(field))
            {
                emitOpeningPointerParen();
            }
            groupedTraverse(leftNode);
            if (!mSymbolEnv.isPointer(field))
            {
                emitClosingPointerParen();
            }
            mOut << ".";
            emitNameOf(field);
        }
        break;

        case TOperator::EOpIndexDirect:
        case TOperator::EOpIndexIndirect:
        {
            TType leftType                   = leftNode.getType();
            const TConstantUnion *constIndex = rightNode.getConstantValue();
            if (leftType.isArray() && IsImage(leftType.getBasicType()) && constIndex != nullptr)
            {
                TIntermBinary *fieldAccess = leftNode.getAsBinaryNode();
                if (fieldAccess && fieldAccess->getOp() == TOperator::EOpIndexDirectStruct)
                {
                    const TField &field =
                        getDirectField(*fieldAccess->getLeft(), *fieldAccess->getRight());
                    groupedTraverse(*fieldAccess->getLeft());
                    mOut << ".";
                    emitNameOf(field);
                    mOut << "_ANGLE_image_element_" << constIndex->getIConst();
                    break;
                }
            }

            groupedTraverse(leftNode);
            mOut << "[";
            // TODO(anglebug.com/42266914): Convert type and bound checks to
            // assertions after AST validation is enabled for MSL translation.
            if (!leftType.isUnsizedArray() && constIndex != nullptr &&
                constIndex->getType() == EbtInt && constIndex->getIConst() >= 0 &&
                constIndex->getIConst() < static_cast<int>(leftType.isArray()
                                                               ? leftType.getOutermostArraySize()
                                                               : leftType.getNominalSize()))
            {
                emitSingleConstant(constIndex);
            }
            else if (leftType.isUnsizedArray())
            {
                // Runtime arrays are emitted as device pointers.  Their valid range is
                // supplied by the host-side SSBO length metadata; clamping against a
                // pointer's .size() is not valid MSL.
                groupedTraverse(rightNode);
            }
            else
            {
                mOut << "ANGLE_int_clamp(";
                groupedTraverse(rightNode);
                mOut << ", 0, ";
                uint32_t maxSize;
                if (leftType.isArray())
                {
                    maxSize = leftType.getOutermostArraySize() - 1;
                }
                else
                {
                    maxSize = leftType.getNominalSize() - 1;
                }
                mOut << maxSize;
                mOut << ")";
            }
            mOut << "]";
            const TField *storageBufferArrayField = getStorageBufferArrayField(leftNode);
            if (leftType.isArray() && storageBufferArrayField)
            {
                mOut << ".value";
            }
        }
        break;

        case TOperator::EOpAddAssign:
        case TOperator::EOpSubAssign:
        case TOperator::EOpMulAssign:
        case TOperator::EOpDivAssign:
        case TOperator::EOpIModAssign:
        case TOperator::EOpBitShiftLeftAssign:
        case TOperator::EOpBitShiftRightAssign:
        case TOperator::EOpVectorTimesScalarAssign:
            // Operators that are emulated with a function do not assign to the left hand side, so
            // decompose e.g. `a %= b` into `a = ANGLE_imod(a, b)`. The left hand side can be
            // emitted twice because SeparateCompoundExpressions has already hoisted out any side
            // effects.
            if (!IsSymbolicOperator(op, binaryNode->getType(), &leftNode.getType(),
                                    &rightNode.getType()))
            {
                groupedTraverse(leftNode);
                mOut << " = ";
            }
            [[fallthrough]];

        default:
        {
            const TType &resultType = binaryNode->getType();
            const TType &leftType   = leftNode.getType();
            const TType &rightType  = rightNode.getType();

            if (IsSymbolicOperator(op, resultType, &leftType, &rightType))
            {
                groupedTraverse(leftNode);
                if (op != TOperator::EOpComma)
                {
                    mOut << " ";
                }
                else
                {
                    emitClosingPointerParen();
                }
                mOut << GetOperatorString(op, resultType, &leftType, &rightType, nullptr) << " ";
                groupedTraverse(rightNode);
            }
            else
            {
                emitClosingPointerParen();
                mOut << GetOperatorString(op, resultType, &leftType, &rightType, nullptr) << "(";
                leftNode.traverse(this);
                mOut << ", ";
                rightNode.traverse(this);
                mOut << ")";
            }
        }
    }

    return false;
}

static bool IsPostfix(TOperator op)
{
    switch (op)
    {
        case TOperator::EOpPostIncrement:
        case TOperator::EOpPostDecrement:
            return true;

        default:
            return false;
    }
}

bool GenMetalTraverser::visitUnary(Visit, TIntermUnary *unaryNode)
{
    const TOperator op      = unaryNode->getOp();
    const TType &resultType = unaryNode->getType();

    TIntermTyped &arg    = *unaryNode->getOperand();
    const TType &argType = arg.getType();

    if (op == TOperator::EOpArrayLength)
    {
        // A Metal device pointer has no length metadata.  The compute runtime supplies a small
        // constant length buffer for every SSBO, exposed as a sibling argument-buffer field.
        TIntermTyped *storageBufferExpression = nullptr;
        const TField *storageBufferField      = nullptr;
        TIntermTyped *storageBufferArrayIndex = nullptr;
        TIntermTyped *cursor                  = &arg;
        while (TIntermBinary *binary = cursor->getAsBinaryNode())
        {
            if (binary->getOp() == TOperator::EOpIndexDirect &&
                binary->getLeft()->getType().isArray())
            {
                const TStructure *indexedStructure = binary->getLeft()->getType().getStruct();
                if (indexedStructure && indexedStructure->isImplementingShaderStorageBlock())
                {
                    // GLES requires constant integral indices for storage-block arrays. Preserve
                    // the selected element so its runtime-array length comes from that binding.
                    storageBufferArrayIndex = binary->getRight();
                }
            }

            if (binary->getOp() == TOperator::EOpIndexDirectStruct ||
                binary->getOp() == TOperator::EOpIndexDirectInterfaceBlock)
            {
                TIntermTyped *left              = binary->getLeft();
                TIntermTyped *right             = binary->getRight();
                const TStructure *leftStructure = left->getType().getStruct();
                if (leftStructure && mPipelineStructs.storageBuffers.matches(*leftStructure))
                {
                    storageBufferExpression = left;
                    storageBufferField      = &getDirectField(*left, *right);
                    break;
                }
            }
            cursor = binary->getLeft();
        }

        if (storageBufferExpression && storageBufferField)
        {
            mOut << "int(*";
            groupedTraverse(*storageBufferExpression);
            mOut << ".";
            emitNameOf(*storageBufferField);
            mOut << "_length";
            if (storageBufferArrayIndex)
            {
                mOut << "[";
                groupedTraverse(*storageBufferArrayIndex);
                mOut << "]";
            }
            mOut << ")";
            return false;
        }

        UNIMPLEMENTED();
        return false;
    }

    if (op == TOperator::EOpIsnan || op == TOperator::EOpIsinf)
    {
        mtl::getTranslatorMetalReflection(&mCompiler)->hasIsnanOrIsinf = true;
    }

    const char *name = GetOperatorString(op, resultType, &argType, nullptr, nullptr);

    if (IsSymbolicOperator(op, resultType, &argType, nullptr))
    {
        const bool postfix = IsPostfix(op);
        if (!postfix)
        {
            mOut << name;
        }
        groupedTraverse(arg);
        if (postfix)
        {
            mOut << name;
        }
    }
    else
    {
        mOut << name << "(";
        arg.traverse(this);
        mOut << ")";
    }

    return false;
}

bool GenMetalTraverser::visitTernary(Visit, TIntermTernary *conditionalNode)
{
    groupedTraverse(*conditionalNode->getCondition());
    mOut << " ? ";
    groupedTraverse(*conditionalNode->getTrueExpression());
    mOut << " : ";
    groupedTraverse(*conditionalNode->getFalseExpression());

    return false;
}

bool GenMetalTraverser::visitIfElse(Visit, TIntermIfElse *ifThenElseNode)
{
    TIntermTyped &condNode = *ifThenElseNode->getCondition();
    TIntermBlock *thenNode = ifThenElseNode->getTrueBlock();
    TIntermBlock *elseNode = ifThenElseNode->getFalseBlock();

    emitIndentation();
    mOut << "if (";
    condNode.traverse(this);
    mOut << ")";

    if (thenNode)
    {
        mOut << "\n";
        thenNode->traverse(this);
    }
    else
    {
        mOut << " {}";
    }

    if (elseNode)
    {
        mOut << "\n";
        emitIndentation();
        mOut << "else\n";
        elseNode->traverse(this);
    }
    else
    {
        // Always emit "else" even when empty block to avoid nested if-stmt issues.
        mOut << " else {}";
    }

    return false;
}

bool GenMetalTraverser::visitSwitch(Visit, TIntermSwitch *switchNode)
{
    emitIndentation();
    mOut << "switch (";
    switchNode->getInit()->traverse(this);
    mOut << ")\n";

    ASSERT(!mParentIsSwitch);
    mParentIsSwitch = true;
    switchNode->getStatementList()->traverse(this);
    mParentIsSwitch = false;

    return false;
}

bool GenMetalTraverser::visitCase(Visit, TIntermCase *caseNode)
{
    emitIndentation();

    if (caseNode->hasCondition())
    {
        TIntermTyped *condExpr = caseNode->getCondition();
        mOut << "case ";
        condExpr->traverse(this);
        mOut << ":";
    }
    else
    {
        mOut << "default:\n";
    }

    return false;
}

void GenMetalTraverser::emitFunctionSignature(const TFunction &func)
{
    const bool isMain = func.isMain();

    emitFunctionReturn(func);

    mOut << " ";
    emitNameOf(func);
    if (isMain)
    {
        mOut << "0";
    }
    mOut << "(";

    bool emitComma = false;
    if (isMain && mCompiler.getShaderType() == GL_COMPUTE_SHADER)
    {
        mOut << "metal::uint3 ANGLE_globalInvocationID [[thread_position_in_grid]], "
             << "metal::uint3 ANGLE_localInvocationID [[thread_position_in_threadgroup]], "
             << "uint32_t ANGLE_localInvocationIndex [[thread_index_in_threadgroup]], "
             << "metal::uint3 ANGLE_workGroupID [[threadgroup_position_in_grid]], "
             << "metal::uint3 ANGLE_numWorkGroups [[threadgroups_per_grid]]";
        emitComma = true;
    }

    const size_t paramCount = func.getParamCount();
    for (size_t i = 0; i < paramCount; ++i)
    {
        if (emitComma)
        {
            mOut << ", ";
        }
        emitComma = true;

        const TVariable &param = *func.getParam(i);
        emitFunctionParameter(func, param);
    }

    if (!isMain && func.symbolType() == SymbolType::UserDefined)
    {
        for (const TVariable *sharedVariable : mSharedVariables)
        {
            if (emitComma)
            {
                mOut << ", ";
            }
            emitComma = true;

            EmitVariableDeclarationConfig evdConfig;
            evdConfig.isParameter            = true;
            evdConfig.disableStructSpecifier = true;
            evdConfig.isPointer              = &kThreadgroupAddressSpace;
            emitVariableDeclaration(VarDecl(*sharedVariable), evdConfig);
        }

        if (mCompiler.getShaderType() == GL_COMPUTE_SHADER)
        {
            for (const char *builtin :
                 {"metal::uint3 ANGLE_globalInvocationID", "metal::uint3 ANGLE_localInvocationID",
                  "uint32_t ANGLE_localInvocationIndex", "metal::uint3 ANGLE_workGroupID",
                  "metal::uint3 ANGLE_numWorkGroups"})
            {
                if (emitComma)
                {
                    mOut << ", ";
                }
                emitComma = true;
                mOut << builtin;
            }
        }
    }

    if (isTraversingVertexMain)
    {
        mOut << " @@XFB-Bindings@@ ";
        mOut << "\n#if TRANSFORM_FEEDBACK_ENABLED\n, device ";
        emitType(func.getReturnType(), EmitTypeConfig{});
        mOut << " * ANGLE_xfbReplayBuffer [[buffer(" << (mSSBOArgumentBufferBindingIndex + 1)
             << ")]]\n#elif TRANSFORM_FEEDBACK_REPLAY\n, device const ";
        emitType(func.getReturnType(), EmitTypeConfig{});
        mOut << " * ANGLE_xfbReplayBuffer [[buffer(" << (mSSBOArgumentBufferBindingIndex + 1)
             << ")]]\n#endif\n";
    }

    mOut << ")";
}

void GenMetalTraverser::emitFunctionReturn(const TFunction &func)
{
    const bool isMain       = func.isMain();
    bool isVertexMain       = false;
    const TType &returnType = func.getReturnType();
    if (isMain)
    {
        if (mCompiler.getShaderType() == GL_COMPUTE_SHADER)
        {
            mOut << "kernel ";
        }
        else
        {
            const TStructure *structure = returnType.getStruct();
            if (mPipelineStructs.fragmentOut.matches(*structure))
            {
                if (mCompiler.isEarlyFragmentTestsSpecified())
                {
                    mOut << "[[early_fragment_tests]]\n";
                }
                mOut << "fragment ";
            }
            else if (mPipelineStructs.vertexOut.matches(*structure))
            {
                ASSERT(structure != nullptr);
                mOut << "vertex __VERTEX_OUT(";
                isVertexMain = true;
            }
            else
            {
                UNIMPLEMENTED();
            }
        }
    }
    emitType(returnType, EmitTypeConfig());
    if (isVertexMain)
    {
        mOut << ") ";
    }
}

void GenMetalTraverser::emitFunctionParameter(const TFunction &func, const TVariable &param)
{
    const bool isMain = func.isMain();

    const TType &type           = param.getType();
    const TStructure *structure = type.getStruct();

    EmitVariableDeclarationConfig evdConfig;
    evdConfig.isParameter            = true;
    evdConfig.disableStructSpecifier = true;  // It's invalid to declare a struct in a function arg.
    evdConfig.isMainParameter        = isMain;
    evdConfig.emitPostQualifier      = isMain;
    evdConfig.isUBO                  = mSymbolEnv.isUBO(param);
    evdConfig.isPointer              = mSymbolEnv.isPointer(param);
    evdConfig.isReference            = mSymbolEnv.isReference(param);
    emitVariableDeclaration(VarDecl(param), evdConfig);

    if (isMain)
    {
        TranslatorMetalReflection *reflection = mtl::getTranslatorMetalReflection(&mCompiler);
        if (structure)
        {
            if (mPipelineStructs.fragmentIn.matches(*structure) ||
                mPipelineStructs.vertexIn.matches(*structure))
            {
                mOut << " [[stage_in]]";
            }
            else if (mPipelineStructs.angleUniforms.matches(*structure))
            {
                mOut << " [[buffer(" << mDriverUniformsBindingIndex << ")]]";
            }
            else if (mPipelineStructs.uniformBuffers.matches(*structure))
            {
                mOut << " [[buffer(" << mUBOArgumentBufferBindingIndex << ")]]";
                reflection->hasUBOs = true;
            }
            else if (mPipelineStructs.storageBuffers.matches(*structure))
            {
                mOut << " [[buffer(" << mSSBOArgumentBufferBindingIndex << ")]]";
                reflection->hasSSBOs = true;
            }
            else if (mPipelineStructs.userUniforms.matches(*structure))
            {
                mOut << " [[buffer(" << mMainUniformBufferIndex << ")]]";
                reflection->addUserUniformBufferBinding(param.name().data(),
                                                        mMainUniformBufferIndex);
                mMainUniformBufferIndex += type.getArraySizeProduct();
            }
            else if (structure->name() == "metal::sampler")
            {
                mOut << " [[sampler(" << (mMainSamplerIndex) << ")]]";
                const std::string originalName =
                    reflection->getOriginalName(param.uniqueId().get());
                reflection->addSamplerBinding(originalName, mMainSamplerIndex);
                mMainSamplerIndex += type.getArraySizeProduct();
            }
        }
        else if (IsSampler(type.getBasicType()))
        {
            mOut << " [[texture(" << (mMainTextureIndex) << ")]]";
            const std::string originalName = reflection->getOriginalName(param.uniqueId().get());
            reflection->addTextureBinding(originalName, mMainTextureIndex);
            mMainTextureIndex += type.getArraySizeProduct();
        }
        else if (Name(param) == Pipeline{Pipeline::Type::InstanceId, nullptr}.getStructInstanceName(
                                    Pipeline::Variant::Modified))
        {
            mOut << " [[instance_id]]";
        }
        else if (Name(param) == kBaseInstanceName)
        {
            mOut << " [[base_instance]]";
        }
    }
}

void GenMetalTraverser::visitFunctionPrototype(TIntermFunctionPrototype *funcProtoNode)
{
    const TFunction &func = *funcProtoNode->getFunction();

    emitIndentation();
    emitFunctionSignature(func);
}

void GenMetalTraverser::emitStorageBufferMatrixHelpers()
{
    if (mStorageBufferMatrixHelpersEmitted)
    {
        return;
    }
    mStorageBufferMatrixHelpersEmitted = true;

    for (const auto &[parent, field] : mStorageBufferMatrixFields)
    {
        TType matrixType(*field->type());
        matrixType.toArrayBaseType();
        size_t majorCount;
        size_t vectorSize;
        size_t vectorStride;
        bool isRowMajor;
        GetStorageBufferMatrixLayout(matrixType, &majorCount, &vectorSize, &vectorStride,
                                     &isRowMajor);
        ASSERT(vectorStride > 0);

        const std::string wrapperName = GetStorageBufferMatrixWrapperName(*parent, *field);
        const std::string helperName  = wrapperName + "_load";
        mOut << "inline ";
        emitType(matrixType, {});
        mOut << " " << helperName << "(" << wrapperName << " storage)\n{\n  ";
        emitType(matrixType, {});
        mOut << " value;\n";
        for (size_t major = 0; major < majorCount; ++major)
        {
            for (size_t minor = 0; minor < vectorSize; ++minor)
            {
                const size_t column = isRowMajor ? minor : major;
                const size_t row    = isRowMajor ? major : minor;
                mOut << "  value[" << column << "][" << row << "] = storage."
                     << (isRowMajor ? "row" : "column") << major << "[" << minor << "];\n";
            }
        }
        mOut << "  return value;\n}\n";

        const std::string storeName = wrapperName + "_store";
        mOut << "inline void " << storeName << "(device " << wrapperName << " *destination, ";
        emitType(matrixType, {});
        mOut << " value)\n{\n";
        for (size_t major = 0; major < majorCount; ++major)
        {
            mOut << "  (*destination)." << (isRowMajor ? "row" : "column") << major
                 << " = metal::packed_float" << vectorSize << "(";
            for (size_t minor = 0; minor < vectorSize; ++minor)
            {
                if (minor != 0)
                {
                    mOut << ", ";
                }
                const size_t column = isRowMajor ? minor : major;
                const size_t row    = isRowMajor ? major : minor;
                mOut << "value[" << column << "][" << row << "]";
            }
            mOut << ");\n";
        }
        mOut << "}\n\n";
    }
}

bool GenMetalTraverser::visitFunctionDefinition(Visit, TIntermFunctionDefinition *funcDefNode)
{
    emitStorageBufferMatrixHelpers();

    const TFunction &func = *funcDefNode->getFunction();
    TIntermBlock &body    = *funcDefNode->getBody();
    if (func.isMain())
    {
        const TType &returnType     = func.getReturnType();
        const TStructure *structure = returnType.getStruct();
        isTraversingVertexMain      = mCompiler.getShaderType() == GL_VERTEX_SHADER && structure &&
                                      mPipelineStructs.vertexOut.matches(*structure);
    }
    emitIndentation();
    emitFunctionSignature(func);
    mOut << "\n";
    mIsComputeMain         = func.isMain() && mCompiler.getShaderType() == GL_COMPUTE_SHADER;
    mInjectSharedVariables = mIsComputeMain;
    body.traverse(this);
    mIsComputeMain = false;
    if (isTraversingVertexMain)
    {
        isTraversingVertexMain = false;
    }
    return false;
}

GenMetalTraverser::FuncToName GenMetalTraverser::BuildFuncToName()
{
    FuncToName map;

    auto putAngle = [&](const char *nameStr) {
        const ImmutableString name(nameStr);
        ASSERT(map.find(name) == map.end());
        map[name] = Name(nameStr, SymbolType::AngleInternal);
    };

    putAngle("texelFetch");
    putAngle("texelFetchOffset");
    putAngle("texture");
    putAngle("texture2D");
    putAngle("texture2DGradEXT");
    putAngle("texture2DLod");
    putAngle("texture2DLodEXT");
    putAngle("texture2DProj");
    putAngle("texture2DProjGradEXT");
    putAngle("texture2DProjLod");
    putAngle("texture2DProjLodEXT");
    putAngle("texture3D");
    putAngle("texture3DLod");
    putAngle("texture3DProj");
    putAngle("texture3DProjLod");
    putAngle("textureCube");
    putAngle("textureCubeGradEXT");
    putAngle("textureCubeLod");
    putAngle("textureCubeLodEXT");
    putAngle("textureGrad");
    putAngle("textureGradOffset");
    putAngle("textureLod");
    putAngle("textureLodOffset");
    putAngle("textureOffset");
    putAngle("textureProj");
    putAngle("textureProjGrad");
    putAngle("textureProjGradOffset");
    putAngle("textureProjLod");
    putAngle("textureProjLodOffset");
    putAngle("textureProjOffset");
    putAngle("textureSize");
    putAngle("textureGather");
    putAngle("textureGatherOffset");
    putAngle("imageSize");
    putAngle("imageLoad");
    putAngle("imageStore");
    putAngle("memoryBarrierImage");

    return map;
}

static const TIntermSymbol *GetRootSymbol(TIntermNode *node)
{
    while (node)
    {
        if (TIntermBinary *binary = node->getAsBinaryNode())
        {
            node = binary->getLeft();
        }
        else if (TIntermSwizzle *swizzle = node->getAsSwizzleNode())
        {
            node = swizzle->getOperand();
        }
        else if (TIntermAggregate *aggregate = node->getAsAggregate())
        {
            const TIntermSymbol *firstSymbol = nullptr;
            for (TIntermNode *child : *aggregate->getSequence())
            {
                const TIntermSymbol *candidate = GetRootSymbol(child);
                if (!candidate)
                {
                    continue;
                }

                const TQualifier qualifier = candidate->variable().getType().getQualifier();
                if (qualifier == TQualifier::EvqBuffer || qualifier == TQualifier::EvqShared)
                {
                    return candidate;
                }
                if (!firstSymbol)
                {
                    firstSymbol = candidate;
                }
            }
            return firstSymbol;
        }
        else
        {
            break;
        }
    }
    return node ? node->getAsSymbolNode() : nullptr;
}

bool GenMetalTraverser::visitAggregate(Visit, TIntermAggregate *aggregateNode)
{
    const TIntermSequence &args = *aggregateNode->getSequence();

    auto emitArgList = [&](const char *open, const char *close,
                           bool appendSharedVariables = false) {
        mOut << open;

        bool emitComma = false;
        for (TIntermNode *arg : args)
        {
            if (emitComma)
            {
                emitClosingPointerParen();
                mOut << ", ";
            }
            emitComma = true;
            arg->traverse(this);
        }

        if (appendSharedVariables)
        {
            for (const TVariable *sharedVariable : mSharedVariables)
            {
                if (emitComma)
                {
                    mOut << ", ";
                }
                emitComma = true;
                if (mIsComputeMain)
                {
                    mOut << "&";
                }
                emitNameOf(*sharedVariable);
            }

            if (mCompiler.getShaderType() == GL_COMPUTE_SHADER)
            {
                for (const char *builtin :
                     {"ANGLE_globalInvocationID", "ANGLE_localInvocationID",
                      "ANGLE_localInvocationIndex", "ANGLE_workGroupID", "ANGLE_numWorkGroups"})
                {
                    if (emitComma)
                    {
                        mOut << ", ";
                    }
                    emitComma = true;
                    mOut << builtin;
                }
            }
        }

        mOut << close;
    };

    const TType &retType = aggregateNode->getType();

    if (aggregateNode->isConstructor())
    {
        const bool isStandalone = getParentNode()->getAsBlock();
        if (isStandalone)
        {
            // Prevent constructor from being interpreted as a declaration by wrapping in parens.
            // This can happen if given something like:
            //      int(symbol); // <- This will be treated like `int symbol;`... don't want that.
            // So instead emit:
            //      (int(symbol));
            mOut << "(";
        }

        const EmitTypeConfig etConfig;

        if (retType.isArray())
        {
            emitType(retType, etConfig);
            emitArgList("{", "}");
        }
        else if (retType.getStruct())
        {
            emitType(retType, etConfig);
            emitArgList("{", "}");
        }
        else if (IsFloatToIntegerConstructor(*aggregateNode))
        {
            mOut << "ANGLE_ftoi<";
            emitType(retType, etConfig);
            mOut << ">";
            emitArgList("(", ")");
        }
        else
        {
            emitType(retType, etConfig);
            emitArgList("(", ")");
        }

        if (isStandalone)
        {
            mOut << ")";
        }

        return false;
    }
    else
    {
        const TOperator op = aggregateNode->getOp();

        const TType *imageArgType = nullptr;
        if (!args.empty())
        {
            TIntermTyped *imageArg = args[0]->getAsTyped();
            ASSERT(imageArg);
            imageArgType = &imageArg->getType();
        }
        TIntermSymbol *imageArgSymbol = args.empty() ? nullptr : args[0]->getAsSymbolNode();
        const bool isImageAtomic      = BuiltInGroup::IsImageAtomic(op);
        const size_t expectedImageArgCount =
            isImageAtomic
                ? (op == TOperator::EOpImageAtomicCompSwap ? 4 : 3)
                : (op == TOperator::EOpImageSize ? 1 : (op == TOperator::EOpImageLoad ? 2 : 3));

        if (BuiltInGroup::IsTextureGather(op) && !BuiltInGroup::IsTextureGatherOffsets(op))
        {
            const TBasicType samplerType =
                aggregateNode->getFunction()->getParam(0)->getType().getBasicType();
            const bool isArray =
                samplerType == EbtSampler2DArray || samplerType == EbtISampler2DArray ||
                samplerType == EbtUSampler2DArray || samplerType == EbtSampler2DArrayShadow;
            const bool isCube = samplerType == EbtSamplerCube || samplerType == EbtISamplerCube ||
                                samplerType == EbtUSamplerCube ||
                                samplerType == EbtSamplerCubeShadow;
            const bool isShadow =
                BuiltInGroup::IsTextureGatherRef(op) || BuiltInGroup::IsTextureGatherOffsetRef(op);
            const bool hasOffset    = BuiltInGroup::IsTextureGatherOffset(op);
            const bool hasComponent = BuiltInGroup::IsTextureGatherComp(op) ||
                                      BuiltInGroup::IsTextureGatherOffsetComp(op);

            mOut << "(";
            args[0]->traverse(this);
            mOut << ").texture->" << (isShadow ? "gather_compare" : "gather") << "(*( ";
            args[0]->traverse(this);
            mOut << ").sampler, ";

            if (isArray)
            {
                mOut << "(";
                args[1]->traverse(this);
                mOut << ").xy, uint32_t(metal::round((";
                args[1]->traverse(this);
                mOut << ").z))";
            }
            else
            {
                args[1]->traverse(this);
            }

            if (isShadow)
            {
                mOut << ", ";
                args[2]->traverse(this);
                if (hasOffset)
                {
                    mOut << ", ";
                    args[3]->traverse(this);
                }
            }
            else
            {
                mOut << ", ";
                if (!isCube)
                {
                    if (hasOffset)
                    {
                        args[2]->traverse(this);
                    }
                    else
                    {
                        mOut << "metal::int2(0)";
                    }
                    mOut << ", ";
                }

                const char *componentName = "x";
                if (hasComponent)
                {
                    const TConstantUnion *componentValue =
                        args[hasOffset ? 3 : 2]->getAsTyped()->getConstantValue();
                    ASSERT(componentValue);
                    switch (componentValue->getIConst())
                    {
                        case 0:
                            componentName = "x";
                            break;
                        case 1:
                            componentName = "y";
                            break;
                        case 2:
                            componentName = "z";
                            break;
                        case 3:
                            componentName = "w";
                            break;
                        default:
                            UNREACHABLE();
                    }
                }
                mOut << "metal::component::" << componentName;
            }

            mOut << ")";
            return false;
        }

        if ((op == TOperator::EOpImageSize || op == TOperator::EOpImageLoad ||
             op == TOperator::EOpImageStore || isImageAtomic) &&
            imageArgType && IsImage2DType(imageArgType->getBasicType()) && !imageArgSymbol &&
            args.size() == expectedImageArgCount + 1)
        {
            const char *functionName =
                isImageAtomic ? GetImageAtomicFunctionName(op)
                              : (op == TOperator::EOpImageSize
                                     ? "ANGLE_imageSize"
                                     : (op == TOperator::EOpImageLoad ? "ANGLE_imageLoad"
                                                                      : "ANGLE_imageStore"));
            mOut << functionName << "(";
            args[0]->traverse(this);
            mOut << ", ";
            args[0]->traverse(this);
            mOut << "_ANGLE_3D, ";
            args.back()->traverse(this);
            for (size_t argIndex = 1; argIndex < expectedImageArgCount; ++argIndex)
            {
                mOut << ", ";
                args[argIndex]->traverse(this);
            }
            mOut << ")";
            return false;
        }

        if (isImageAtomic)
        {
            mOut << GetImageAtomicFunctionName(op);
            emitArgList("(", ")");
            return false;
        }

        if (op == TOperator::EOpImageSize && imageArgType &&
            IsImageCubeType(imageArgType->getBasicType()) &&
            mCompileOptions.emulateR32fImageAtomicExchange)
        {
            mOut << "ANGLE_imageSize(";
            args[0]->traverse(this);
            mOut << ").xy";
            return false;
        }

        switch (op)
        {
            case TOperator::EOpAtomicAdd:
            case TOperator::EOpAtomicMin:
            case TOperator::EOpAtomicMax:
            case TOperator::EOpAtomicAnd:
            case TOperator::EOpAtomicOr:
            case TOperator::EOpAtomicXor:
            case TOperator::EOpAtomicExchange:
            case TOperator::EOpAtomicCompSwap:
            {
                const bool isCompareSwap = op == TOperator::EOpAtomicCompSwap;
                ASSERT(args.size() == (isCompareSwap ? 3u : 2u));
                TIntermTyped *memory = args[0]->getAsTyped();
                ASSERT(memory);
                const TIntermSymbol *rootSymbol = GetRootSymbol(args[0]);
                ASSERT(rootSymbol);

                const TQualifier qualifier = rootSymbol->variable().getType().getQualifier();
                const char *addressSpace;
                const char *atomicFunction;
                switch (qualifier)
                {
                    case TQualifier::EvqShared:
                        addressSpace = "threadgroup";
                        break;
                    case TQualifier::EvqBuffer:
                        addressSpace = "device";
                        break;
                    default:
                        UNIMPLEMENTED();
                        addressSpace = "device";
                        break;
                }

                switch (op)
                {
                    case TOperator::EOpAtomicAdd:
                        atomicFunction = "metal::atomic_fetch_add_explicit";
                        break;
                    case TOperator::EOpAtomicMin:
                        atomicFunction = "metal::atomic_fetch_min_explicit";
                        break;
                    case TOperator::EOpAtomicMax:
                        atomicFunction = "metal::atomic_fetch_max_explicit";
                        break;
                    case TOperator::EOpAtomicAnd:
                        atomicFunction = "metal::atomic_fetch_and_explicit";
                        break;
                    case TOperator::EOpAtomicOr:
                        atomicFunction = "metal::atomic_fetch_or_explicit";
                        break;
                    case TOperator::EOpAtomicXor:
                        atomicFunction = "metal::atomic_fetch_xor_explicit";
                        break;
                    case TOperator::EOpAtomicExchange:
                        atomicFunction = "metal::atomic_exchange_explicit";
                        break;
                    case TOperator::EOpAtomicCompSwap:
                        atomicFunction = "ANGLE_atomicCompSwap";
                        break;
                    default:
                        UNREACHABLE();
                        atomicFunction = "";
                        break;
                }

                const char *atomicType =
                    memory->getType().isSignedInt() ? "metal::atomic_int" : "metal::atomic_uint";
                auto emitAtomicMemoryPointer = [&]() {
                    TIntermAggregate *memoryReference = args[0]->getAsAggregate();
                    const TFunction *referenceFunction =
                        memoryReference ? memoryReference->getFunction() : nullptr;
                    const bool isElementReference =
                        referenceFunction && referenceFunction->name() == "elem_ref";
                    const bool isSwizzleReference =
                        referenceFunction && referenceFunction->name() == "swizzle_ref";
                    if (!isElementReference && !isSwizzleReference)
                    {
                        mOut << "reinterpret_cast<" << addressSpace << " " << atomicType << " *>(&";
                        args[0]->traverse(this);
                        mOut << ")";
                        return;
                    }

                    const TIntermSequence &referenceArgs = *memoryReference->getSequence();
                    ASSERT(referenceArgs.size() == 2);
                    TIntermTyped *vector = referenceArgs[0]->getAsTyped();
                    ASSERT(vector && vector->getType().isVector());
                    mOut << "(reinterpret_cast<" << addressSpace << " " << atomicType << " *>(&";
                    referenceArgs[0]->traverse(this);
                    mOut << ") + ";
                    if (isElementReference)
                    {
                        mOut << "metal::clamp(int(";
                    }
                    referenceArgs[1]->traverse(this);
                    if (isElementReference)
                    {
                        mOut << "), 0, " << vector->getType().getNominalSize() - 1 << ")";
                    }
                    mOut << ")";
                };
                mOut << atomicFunction << "(";
                emitAtomicMemoryPointer();
                mOut << ", ";
                args[1]->traverse(this);
                if (isCompareSwap)
                {
                    mOut << ", ";
                    args[2]->traverse(this);
                }
                else
                {
                    mOut << ", metal::memory_order_relaxed";
                }
                mOut << ")";
                return false;
            }

            case TOperator::EOpCallFunctionInAST:
            case TOperator::EOpCallInternalRawFunction:
            {
                const TFunction &func = *aggregateNode->getFunction();
                emitNameOf(func);
                //'@' symbol in name specifices a macro substitution marker.
                if (!func.name().contains("@"))
                {
                    emitArgList("(", ")", func.symbolType() == SymbolType::UserDefined);
                }
                else
                {
                    mTemporarilyDisableSemicolon =
                        true;  // Disable semicolon for macro substitution.
                }
                return false;
            }

            default:
            {
                auto getArgType = [&](size_t index) -> const TType * {
                    if (index < args.size())
                    {
                        TIntermTyped *arg = args[index]->getAsTyped();
                        ASSERT(arg);
                        return &arg->getType();
                    }
                    return nullptr;
                };

                const TType *argType0 = getArgType(0);
                const TType *argType1 = getArgType(1);
                const TType *argType2 = getArgType(2);

                const char *opName = GetOperatorString(op, retType, argType0, argType1, argType2);

                if (IsSymbolicOperator(op, retType, argType0, argType1))
                {
                    switch (args.size())
                    {
                        case 1:
                        {
                            TIntermNode &operandNode = *aggregateNode->getChildNode(0);
                            if (IsPostfix(op))
                            {
                                mOut << opName;
                                groupedTraverse(operandNode);
                            }
                            else
                            {
                                groupedTraverse(operandNode);
                                mOut << opName;
                            }
                            return false;
                        }

                        case 2:
                        {
                            TIntermNode &leftNode  = *aggregateNode->getChildNode(0);
                            TIntermNode &rightNode = *aggregateNode->getChildNode(1);
                            groupedTraverse(leftNode);
                            mOut << " " << opName << " ";
                            groupedTraverse(rightNode);
                            return false;
                        }

                        default:
                            UNREACHABLE();
                            return false;
                    }
                }
                else if (opName == nullptr)
                {
                    const TFunction &func = *aggregateNode->getFunction();
                    auto it               = mFuncToName.find(func.name());
                    ASSERT(it != mFuncToName.end());
                    EmitName(mOut, it->second, kUserVariableNamePrefix);
                    emitArgList("(", ")");
                    return false;
                }
                else
                {
                    mOut << opName;
                    emitArgList("(", ")");
                    return false;
                }
            }
        }
    }
}

void GenMetalTraverser::emitOpenBrace()
{
    ASSERT(mIndentLevel >= 0);

    emitIndentation();
    mOut << "{\n";
    ++mIndentLevel;
}

void GenMetalTraverser::emitCloseBrace()
{
    ASSERT(mIndentLevel >= 1);

    --mIndentLevel;
    emitIndentation();
    mOut << "}";
}

static bool RequiresSemicolonTerminator(TIntermNode &node)
{
    if (node.getAsBlock())
    {
        return false;
    }
    if (node.getAsLoopNode())
    {
        return false;
    }
    if (node.getAsSwitchNode())
    {
        return false;
    }
    if (node.getAsIfElseNode())
    {
        return false;
    }
    if (node.getAsFunctionDefinition())
    {
        return false;
    }
    if (node.getAsCaseNode())
    {
        return false;
    }

    return true;
}

static bool NewlinePad(TIntermNode &node)
{
    if (node.getAsFunctionDefinition())
    {
        return true;
    }
    if (TIntermDeclaration *declNode = node.getAsDeclarationNode())
    {
        ASSERT(declNode->getChildCount() == 1);
        TIntermNode &childNode = *declNode->getChildNode(0);
        if (TIntermSymbol *symbolNode = childNode.getAsSymbolNode())
        {
            const TVariable &var = symbolNode->variable();
            return var.getType().isStructSpecifier();
        }
        return false;
    }
    return false;
}

bool GenMetalTraverser::visitBlock(Visit, TIntermBlock *blockNode)
{
    ASSERT(mIndentLevel >= -1);
    const bool isGlobalScope  = mIndentLevel == -1;
    const bool parentIsSwitch = mParentIsSwitch;
    mParentIsSwitch           = false;

    if (isGlobalScope)
    {
        ++mIndentLevel;
    }
    else
    {
        emitOpenBrace();
        if (mInjectSharedVariables)
        {
            mInjectSharedVariables = false;
            for (const TVariable *sharedVariable : mSharedVariables)
            {
                emitIndentation();
                EmitVariableDeclarationConfig evdConfig;
                evdConfig.disableStructSpecifier = true;
                evdConfig.storageAddressSpace    = &kThreadgroupAddressSpace;
                emitVariableDeclaration(VarDecl(*sharedVariable), evdConfig);
                mOut << ";\n";
            }
        }
        if (parentIsSwitch)
        {
            ++mIndentLevel;
        }
    }

    TIntermNode *prevStmtNode = nullptr;

    const size_t stmtCount = blockNode->getChildCount();
    for (size_t i = 0; i < stmtCount; ++i)
    {
        TIntermNode &stmtNode = *blockNode->getChildNode(i);

        if (isGlobalScope && prevStmtNode && (NewlinePad(*prevStmtNode) || NewlinePad(stmtNode)))
        {
            mOut << "\n";
        }
        const bool isCase = stmtNode.getAsCaseNode();
        mIndentLevel -= isCase;
        emitIndentation();
        mIndentLevel += isCase;
        stmtNode.traverse(this);
        if (RequiresSemicolonTerminator(stmtNode) && !mTemporarilyDisableSemicolon)
        {
            mOut << ";";
        }
        mTemporarilyDisableSemicolon = false;
        mOut << "\n";

        prevStmtNode = &stmtNode;
    }

    if (isGlobalScope)
    {
        ASSERT(mIndentLevel == 0);
        --mIndentLevel;
    }
    else
    {
        if (parentIsSwitch)
        {
            ASSERT(mIndentLevel >= 1);
            --mIndentLevel;
        }
        emitCloseBrace();
        mParentIsSwitch = parentIsSwitch;
    }

    return false;
}

bool GenMetalTraverser::visitGlobalQualifierDeclaration(Visit, TIntermGlobalQualifierDeclaration *)
{
    return false;
}

bool GenMetalTraverser::visitDeclaration(Visit, TIntermDeclaration *declNode)
{
    ASSERT(declNode->getChildCount() == 1);
    TIntermNode &node = *declNode->getChildNode(0);

    EmitVariableDeclarationConfig evdConfig;

    if (TIntermSymbol *symbolNode = node.getAsSymbolNode())
    {
        const TVariable &var = symbolNode->variable();
        if (var.getType().getQualifier() == TQualifier::EvqShared)
        {
            mSharedVariables.push_back(&var);
            return false;
        }
        emitVariableDeclaration(VarDecl(var), evdConfig);
        if (var.getType().isArray() && var.getType().getQualifier() == EvqTemporary)
        {
            // The translator frontend injects a loop-based init for user arrays when the source
            // shader is using ESSL 1.00. Some Metal drivers may fail to access elements of such
            // arrays at runtime depending on the array size. An empty literal initializer added
            // to the generated MSL bypasses the issue. The frontend may be further optimized to
            // skip the loop-based init when targeting MSL.
            mOut << "{}";
        }
    }
    else if (TIntermBinary *initNode = node.getAsBinaryNode())
    {
        ASSERT(initNode->getOp() == TOperator::EOpInitialize);
        TIntermSymbol *leftSymbolNode = initNode->getLeft()->getAsSymbolNode();
        TIntermTyped *valueNode       = initNode->getRight()->getAsTyped();
        ASSERT(leftSymbolNode && valueNode);

        if (getRootNode() == getParentBlock())
        {
            // DeferGlobalInitializers should have turned non-const global initializers into
            // deferred initializers. Note that variables marked as EvqGlobal can be treated as
            // EvqConst in some ANGLE code but not actually have their qualifier actually changed to
            // EvqConst. Thus just assume all EvqGlobal are actually EvqConst for all code run after
            // DeferGlobalInitializers.
            mOut << "constant ";
        }

        const TVariable &var = leftSymbolNode->variable();
        const Name varName(var);

        if (ExpressionContainsName(varName, *valueNode))
        {
            mRenamedSymbols[&var] = mIdGen.createNewName(varName);
        }

        emitVariableDeclaration(VarDecl(var), evdConfig);
        mOut << " = ";
        groupedTraverse(*valueNode);
    }
    else
    {
        UNREACHABLE();
    }

    return false;
}

bool GenMetalTraverser::visitLoop(Visit, TIntermLoop *loopNode)
{
    const TLoopType loopType = loopNode->getType();

    switch (loopType)
    {
        case TLoopType::ELoopFor:
            return visitForLoop(loopNode);
        case TLoopType::ELoopWhile:
            return visitWhileLoop(loopNode);
        case TLoopType::ELoopDoWhile:
            return visitDoWhileLoop(loopNode);
    }
}

bool GenMetalTraverser::visitForLoop(TIntermLoop *loopNode)
{
    ASSERT(loopNode->getType() == TLoopType::ELoopFor);

    TIntermNode *initNode  = loopNode->getInit();
    TIntermTyped *condNode = loopNode->getCondition();
    TIntermTyped *exprNode = loopNode->getExpression();

    mOut << "for (";

    if (initNode)
    {
        initNode->traverse(this);
    }
    else
    {
        mOut << " ";
    }

    mOut << "; ";

    if (condNode)
    {
        condNode->traverse(this);
    }

    mOut << "; ";

    if (exprNode)
    {
        exprNode->traverse(this);
    }

    mOut << ")\n";

    emitLoopBody(loopNode->getBody());

    return false;
}

bool GenMetalTraverser::visitWhileLoop(TIntermLoop *loopNode)
{
    ASSERT(loopNode->getType() == TLoopType::ELoopWhile);

    TIntermNode *initNode  = loopNode->getInit();
    TIntermTyped *condNode = loopNode->getCondition();
    TIntermTyped *exprNode = loopNode->getExpression();
    ASSERT(condNode);
    ASSERT(!initNode && !exprNode);

    emitIndentation();
    mOut << "while (";
    condNode->traverse(this);
    mOut << ")\n";
    emitLoopBody(loopNode->getBody());

    return false;
}

bool GenMetalTraverser::visitDoWhileLoop(TIntermLoop *loopNode)
{
    ASSERT(loopNode->getType() == TLoopType::ELoopDoWhile);

    TIntermNode *initNode  = loopNode->getInit();
    TIntermTyped *condNode = loopNode->getCondition();
    TIntermTyped *exprNode = loopNode->getExpression();
    ASSERT(condNode);
    ASSERT(!initNode && !exprNode);

    emitIndentation();
    mOut << "do\n";
    emitLoopBody(loopNode->getBody());
    mOut << "\n";
    emitIndentation();
    mOut << "while (";
    condNode->traverse(this);
    mOut << ");";

    return false;
}

bool GenMetalTraverser::visitBranch(Visit, TIntermBranch *branchNode)
{
    const TOperator flowOp = branchNode->getFlowOp();
    TIntermTyped *exprNode = branchNode->getExpression();

    emitIndentation();

    switch (flowOp)
    {
        case TOperator::EOpKill:
        {
            ASSERT(exprNode == nullptr);
            mOut << "metal::discard_fragment()";
        }
        break;

        case TOperator::EOpReturn:
        {
            if (isTraversingVertexMain)
            {
                mOut << "#if TRANSFORM_FEEDBACK_ENABLED\n";
                emitIndentation();
                mOut << "return;\n";
                emitIndentation();
                mOut << "#else\n";
                emitIndentation();
            }
            mOut << "return";
            if (exprNode)
            {
                mOut << " ";
                exprNode->traverse(this);
                mOut << ";";
            }
            if (isTraversingVertexMain)
            {
                mOut << "\n";
                emitIndentation();
                mOut << "#endif\n";
                mTemporarilyDisableSemicolon = true;
            }
        }
        break;

        case TOperator::EOpBreak:
        {
            ASSERT(exprNode == nullptr);
            mOut << "break";
        }
        break;

        case TOperator::EOpContinue:
        {
            ASSERT(exprNode == nullptr);
            mOut << "continue";
        }
        break;

        default:
        {
            UNREACHABLE();
        }
    }

    return false;
}

static size_t emitMetalCallCount = 0;

bool sh::EmitMetal(TCompiler &compiler,
                   TIntermBlock &root,
                   IdGen &idGen,
                   const PipelineStructs &pipelineStructs,
                   SymbolEnv &symbolEnv,
                   const ProgramPreludeConfig &ppc,
                   const ShCompileOptions &compileOptions)
{
    TInfoSinkBase &out = compiler.getInfoSink().obj;

    {
        ++emitMetalCallCount;
        std::string filenameProto = angle::GetEnvironmentVar("GMD_FIXED_EMIT");
        if (!filenameProto.empty())
        {
            if (filenameProto != "/dev/null")
            {
                auto tryOpen = [&](char const *ext) {
                    auto filename = filenameProto;
                    filename += std::to_string(emitMetalCallCount);
                    filename += ".";
                    filename += ext;
                    return fopen(filename.c_str(), "rb");
                };
                FILE *file = tryOpen("metal");
                if (!file)
                {
                    file = tryOpen("cpp");
                }
                ASSERT(file);

                fseek(file, 0, SEEK_END);
                size_t fileSize = ftell(file);
                fseek(file, 0, SEEK_SET);

                std::vector<char> buff;
                buff.resize(fileSize + 1);
                ANGLE_UNSAFE_TODO(fread(buff.data(), fileSize, 1, file));
                buff.back() = '\0';

                fclose(file);

                out << buff.data();
            }

            return true;
        }
    }

    out << "\n\n";

    if (!EmitProgramPrelude(root, out, ppc))
    {
        return false;
    }

    {
#if defined(ANGLE_ENABLE_ASSERTS)
        DebugSink outWrapper(out, angle::GetBoolEnvironmentVar("GMD_STDOUT"));
        outWrapper.watch(angle::GetEnvironmentVar("GMD_WATCH_STRING"));
#else
        TInfoSinkBase &outWrapper = out;
#endif
        GenMetalTraverser gen(compiler, outWrapper, idGen, pipelineStructs, symbolEnv, ppc,
                              compileOptions);
        root.traverse(&gen);
    }

    out << "\n";

    return true;
}
