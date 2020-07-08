// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "open3d/core/TensorList.h"

#include <vector>

#include "tests/UnitTest.h"
#include "tests/core/CoreTest.h"

namespace open3d {
namespace tests {

class TensorListPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(TensorList,
                         TensorListPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

TEST_P(TensorListPermuteDevices, EmptyConstructor) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    // TensorList allows 0-sized and scalar {} element_shape.
    for (const core::SizeVector& element_shape : std::vector<core::SizeVector>{
                 {},   // Scalar {} element_shape is fine.
                 {0},  // 0-sized element_shape is fine.
                 {1},  // This is different from {}.
                 {0, 0},
                 {0, 1},
                 {1, 0},
                 {2, 3},
         }) {
        core::TensorList tl(element_shape, dtype, device);
        EXPECT_EQ(tl.GetElementShape(), element_shape);
        EXPECT_EQ(tl.GetDtype(), dtype);
        EXPECT_EQ(tl.GetDevice(), device);
    }

    // TensorList does not allow negative element_shape.
    EXPECT_ANY_THROW(core::TensorList({0, -1}, dtype, device));
    EXPECT_ANY_THROW(core::TensorList({-1, -1}, dtype, device));
}

TEST_P(TensorListPermuteDevices, ConstructFromTensorVector) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    core::Tensor t0 = core::Tensor::Ones({2, 3}, dtype, device) * 0.;
    core::Tensor t1 = core::Tensor::Ones({2, 3}, dtype, device) * 1.;
    core::Tensor t2 = core::Tensor::Ones({2, 3}, dtype, device) * 2.;
    core::TensorList tl(std::vector<core::Tensor>({t0, t1, t2}));

    // Check tensor list.
    core::SizeVector full_shape({3, 2, 3});
    EXPECT_EQ(tl.AsTensor().GetShape(), full_shape);
    EXPECT_EQ(tl.GetSize(), 3);
    EXPECT_EQ(tl.GetReservedSize(), 8);

    // Values should be copied. IsClose also ensures the same dtype and device.
    EXPECT_TRUE(tl[0].AllClose(t0));
    EXPECT_TRUE(tl[1].AllClose(t1));
    EXPECT_TRUE(tl[2].AllClose(t2));
    EXPECT_FALSE(tl[0].IsSame(t0));
    EXPECT_FALSE(tl[1].IsSame(t1));
    EXPECT_FALSE(tl[2].IsSame(t2));

    // Device mismatch.
    core::Tensor t3 = core::Tensor::Ones({2, 3}, dtype, core::Device("CPU:0"));
    core::Tensor t4 = core::Tensor::Ones({2, 3}, dtype, device);
    if (t3.GetDevice() != t4.GetDevice()) {
        // This tests only fires when CUDA is available.
        EXPECT_ANY_THROW(core::TensorList(std::vector<core::Tensor>({t3, t4})));
    }

    // Shape mismatch.
    core::Tensor t5 = core::Tensor::Ones({2, 3}, core::Dtype::Float32, device);
    core::Tensor t6 = core::Tensor::Ones({2, 3}, core::Dtype::Float64, device);
    EXPECT_ANY_THROW(core::TensorList(std::vector<core::Tensor>({t5, t6})));
}

TEST_P(TensorListPermuteDevices, ConstructFromTensors) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    core::Tensor t0 = core::Tensor::Ones({2, 3}, dtype, device) * 0.;
    core::Tensor t1 = core::Tensor::Ones({2, 3}, dtype, device) * 1.;
    core::Tensor t2 = core::Tensor::Ones({2, 3}, dtype, device) * 2.;
    std::vector<core::Tensor> tensors({t0, t1, t2});

    for (const core::TensorList& tl : std::vector<core::TensorList>({
                 core::TensorList(tensors),
                 core::TensorList(tensors.begin(), tensors.end()),
                 core::TensorList({t0, t1, t2}),
         })) {
        core::SizeVector full_shape({3, 2, 3});
        EXPECT_EQ(tl.AsTensor().GetShape(), full_shape);
        EXPECT_EQ(tl.GetSize(), 3);
        EXPECT_EQ(tl.GetReservedSize(), 8);
        // Values are the same.
        EXPECT_TRUE(tl[0].AllClose(t0));
        EXPECT_TRUE(tl[1].AllClose(t1));
        EXPECT_TRUE(tl[2].AllClose(t2));
        // Tensors are copied.
        EXPECT_FALSE(tl[0].IsSame(t0));
        EXPECT_FALSE(tl[1].IsSame(t1));
        EXPECT_FALSE(tl[2].IsSame(t2));
    }

    // Device mismatch.
    core::Tensor t3 = core::Tensor::Ones({2, 3}, dtype, core::Device("CPU:0"));
    core::Tensor t4 = core::Tensor::Ones({2, 3}, dtype, device);
    if (t3.GetDevice() != t4.GetDevice()) {
        // This tests only fires when CUDA is available.
        EXPECT_ANY_THROW(core::TensorList(std::vector<core::Tensor>({t3, t4})));
    }

    // Shape mismatch.
    core::Tensor t5 = core::Tensor::Ones({2, 3}, core::Dtype::Float32, device);
    core::Tensor t6 = core::Tensor::Ones({2, 3}, core::Dtype::Float64, device);
    EXPECT_ANY_THROW(core::TensorList(std::vector<core::Tensor>({t5, t6})));
}

TEST_P(TensorListPermuteDevices, FromTensor) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;
    core::Tensor t = core::Tensor::Ones({3, 4, 5}, dtype, device);

    // Copyied tensor.
    core::TensorList tl = core::TensorList::FromTensor(t);
    EXPECT_EQ(tl.GetElementShape(), core::SizeVector({4, 5}));
    EXPECT_EQ(tl.GetSize(), 3);
    EXPECT_EQ(tl.GetReservedSize(), 8);
    EXPECT_TRUE(tl.AsTensor().AllClose(t));
    EXPECT_FALSE(tl.AsTensor().IsSame(t));

    // Inplace tensor.
    core::TensorList tl_inplace = core::TensorList::FromTensor(t, true);
    EXPECT_EQ(tl_inplace.GetElementShape(), core::SizeVector({4, 5}));
    EXPECT_EQ(tl_inplace.GetSize(), 3);
    EXPECT_EQ(tl_inplace.GetReservedSize(), 3);
    EXPECT_TRUE(tl_inplace.AsTensor().AllClose(t));
    EXPECT_TRUE(tl_inplace.AsTensor().IsSame(t));
}

TEST_P(TensorListPermuteDevices, CopyConstructor) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;
    core::Tensor t = core::Tensor::Ones({3, 4, 5}, dtype, device);

    core::TensorList tl = core::TensorList::FromTensor(t, false);
    core::TensorList tl_copy(tl);
    EXPECT_TRUE(tl.AsTensor().IsSame(tl_copy.AsTensor()));
}

TEST_P(TensorListPermuteDevices, MoveConstructor) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;
    core::Tensor t = core::Tensor::Ones({3, 4, 5}, dtype, device);

    auto create_tl = [&t]() {
        return core::TensorList::FromTensor(t, /*inplace=*/true);
    };
    core::TensorList tl(create_tl());
    EXPECT_TRUE(tl.AsTensor().IsSame(t));
}

TEST_P(TensorListPermuteDevices, Resize) {
    core::Device device = GetParam();

    core::Tensor t0(std::vector<float>(2 * 3, 0), {2, 3}, core::Dtype::Float32,
                    device);
    core::Tensor t1(std::vector<float>(2 * 3, 1), {2, 3}, core::Dtype::Float32,
                    device);
    core::Tensor t2(std::vector<float>(2 * 3, 2), {2, 3}, core::Dtype::Float32,
                    device);

    std::vector<core::Tensor> tensors = {t0, t1, t2};
    core::TensorList tensor_list(tensors);
    EXPECT_EQ(tensor_list.GetSize(), 3);
    EXPECT_EQ(tensor_list.GetReservedSize(), 8);
    EXPECT_EQ(tensor_list.AsTensor().ToFlatVector<float>(),
              std::vector<float>(
                      {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2}));

    tensor_list.Resize(5);
    EXPECT_EQ(tensor_list.GetSize(), 5);
    EXPECT_EQ(tensor_list.GetReservedSize(), 16);
    EXPECT_EQ(
            tensor_list.AsTensor().ToFlatVector<float>(),
            std::vector<float>({0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2,
                                2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));

    tensor_list.Resize(2);
    EXPECT_EQ(tensor_list.GetSize(), 2);
    EXPECT_EQ(tensor_list.GetReservedSize(), 16);
    EXPECT_EQ(tensor_list.AsTensor().ToFlatVector<float>(),
              std::vector<float>({0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1}));
}

TEST_P(TensorListPermuteDevices, PushBack) {
    core::Device device = GetParam();

    core::Tensor t0(std::vector<float>(2 * 3, 0), {2, 3}, core::Dtype::Float32,
                    device);
    core::Tensor t1(std::vector<float>(2 * 3, 1), {2, 3}, core::Dtype::Float32,
                    device);
    core::Tensor t2(std::vector<float>(2 * 3, 2), {2, 3}, core::Dtype::Float32,
                    device);

    core::TensorList tensor_list({2, 3}, core::Dtype::Float32);
    EXPECT_EQ(tensor_list.GetSize(), 0);
    EXPECT_EQ(tensor_list.GetReservedSize(), 1);

    tensor_list.PushBack(t0);
    EXPECT_EQ(tensor_list.GetSize(), 1);
    EXPECT_EQ(tensor_list.GetReservedSize(), 2);
    EXPECT_EQ(tensor_list.AsTensor().ToFlatVector<float>(),
              std::vector<float>({0, 0, 0, 0, 0, 0}));

    tensor_list.PushBack(t1);
    EXPECT_EQ(tensor_list.GetSize(), 2);
    EXPECT_EQ(tensor_list.GetReservedSize(), 4);
    EXPECT_EQ(tensor_list.AsTensor().ToFlatVector<float>(),
              std::vector<float>({0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1}));

    tensor_list.PushBack(t2);
    EXPECT_EQ(tensor_list.GetSize(), 3);
    EXPECT_EQ(tensor_list.GetReservedSize(), 8);
    EXPECT_EQ(tensor_list.AsTensor().ToFlatVector<float>(),
              std::vector<float>(
                      {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2}));
}

TEST_P(TensorListPermuteDevices, AccessOperator) {
    core::Device device = GetParam();

    core::Tensor t0(std::vector<float>(2 * 3, 0), {2, 3}, core::Dtype::Float32,
                    device);
    core::Tensor t1(std::vector<float>(2 * 3, 1), {2, 3}, core::Dtype::Float32,
                    device);
    core::Tensor t2(std::vector<float>(2 * 3, 2), {2, 3}, core::Dtype::Float32,
                    device);

    std::vector<core::Tensor> tensors = {t0, t1, t2};
    core::TensorList tensor_list(tensors);

    EXPECT_EQ(tensor_list.GetSize(), 3);
    EXPECT_EQ(tensor_list[0].ToFlatVector<float>(), t0.ToFlatVector<float>());
    EXPECT_EQ(tensor_list[1].ToFlatVector<float>(), t1.ToFlatVector<float>());
    EXPECT_EQ(tensor_list[2].ToFlatVector<float>(), t2.ToFlatVector<float>());

    tensor_list[0] = t2;
    tensor_list[1] = t1;
    tensor_list[2] = t0;

    EXPECT_EQ(tensor_list.AsTensor().ToFlatVector<float>(),
              std::vector<float>(
                      {2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0}));
}

TEST_P(TensorListPermuteDevices, Slice) {
    core::Device device = GetParam();

    core::Tensor t0(std::vector<float>(2 * 3, 0), {2, 3}, core::Dtype::Float32,
                    device);
    core::Tensor t1(std::vector<float>(2 * 3, 1), {2, 3}, core::Dtype::Float32,
                    device);
    core::Tensor t2(std::vector<float>(2 * 3, 2), {2, 3}, core::Dtype::Float32,
                    device);
    core::Tensor t3(std::vector<float>(2 * 3, 3), {2, 3}, core::Dtype::Float32,
                    device);

    std::vector<core::Tensor> tensors = {t0, t1, t2, t3};
    core::TensorList tensor_list(tensors);

    core::Tensor tensor = tensor_list.AsTensor().Slice(0, 0, 3, 2);
    EXPECT_EQ(tensor.ToFlatVector<float>(),
              std::vector<float>({0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2}));
}

TEST_P(TensorListPermuteDevices, IndexGet) {
    core::Device device = GetParam();

    core::Tensor t0(std::vector<float>(2 * 3, 0), {2, 3}, core::Dtype::Float32,
                    device);
    core::Tensor t1(std::vector<float>(2 * 3, 1), {2, 3}, core::Dtype::Float32,
                    device);
    core::Tensor t2(std::vector<float>(2 * 3, 2), {2, 3}, core::Dtype::Float32,
                    device);
    core::Tensor t3(std::vector<float>(2 * 3, 3), {2, 3}, core::Dtype::Float32,
                    device);

    std::vector<core::Tensor> tensors = {t0, t1, t2, t3};
    core::TensorList tensor_list(tensors);

    std::vector<core::Tensor> indices = {core::Tensor(
            std::vector<int64_t>({0, -1, 2}), {3}, core::Dtype::Int64, device)};
    core::Tensor tensor = tensor_list.AsTensor().IndexGet(indices);
    EXPECT_EQ(tensor.ToFlatVector<float>(),
              std::vector<float>(
                      {0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2}));
}

TEST_P(TensorListPermuteDevices, Concatenate) {
    core::Device device = GetParam();

    core::Tensor t0(std::vector<float>(2 * 3, 0), {2, 3}, core::Dtype::Float32,
                    device);
    std::vector<core::Tensor> tensors0 = {t0};

    core::Tensor t1(std::vector<float>(2 * 3, 1), {2, 3}, core::Dtype::Float32,
                    device);
    core::Tensor t2(std::vector<float>(2 * 3, 2), {2, 3}, core::Dtype::Float32,
                    device);
    core::Tensor t3(std::vector<float>(2 * 3, 3), {2, 3}, core::Dtype::Float32,
                    device);
    std::vector<core::Tensor> tensors1 = {t1, t2, t3};

    core::TensorList tensor_list0(tensors0);
    core::TensorList tensor_list1(tensors1);

    core::TensorList tensor_list2 = tensor_list0 + tensor_list1;
    EXPECT_EQ(tensor_list2.GetSize(), 4);
    EXPECT_EQ(tensor_list2.GetReservedSize(), 8);
    EXPECT_EQ(tensor_list2.AsTensor().ToFlatVector<float>(),
              std::vector<float>({0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                                  2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3}));

    core::TensorList tensor_list3 =
            core::TensorList::Concatenate(tensor_list1, tensor_list0);
    EXPECT_EQ(tensor_list3.GetSize(), 4);
    EXPECT_EQ(tensor_list3.GetReservedSize(), 8);
    EXPECT_EQ(tensor_list3.AsTensor().ToFlatVector<float>(),
              std::vector<float>({1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                                  3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0}));
}

TEST_P(TensorListPermuteDevices, Extend) {
    core::Device device = GetParam();

    core::Tensor t0(std::vector<float>(2 * 3, 0), {2, 3}, core::Dtype::Float32,
                    device);
    std::vector<core::Tensor> tensors0 = {t0};

    core::Tensor t1(std::vector<float>(2 * 3, 1), {2, 3}, core::Dtype::Float32,
                    device);
    core::Tensor t2(std::vector<float>(2 * 3, 2), {2, 3}, core::Dtype::Float32,
                    device);
    core::Tensor t3(std::vector<float>(2 * 3, 3), {2, 3}, core::Dtype::Float32,
                    device);
    std::vector<core::Tensor> tensors1 = {t1, t2, t3};

    core::TensorList tensor_list0(tensors0);
    core::TensorList tensor_list1(tensors1);

    tensor_list0.Extend(tensor_list1);
    EXPECT_EQ(tensor_list0.GetSize(), 4);
    EXPECT_EQ(tensor_list0.GetReservedSize(), 8);
    EXPECT_EQ(tensor_list0.AsTensor().ToFlatVector<float>(),
              std::vector<float>({0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                                  2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3}));

    tensor_list1 += tensor_list1;
    EXPECT_EQ(tensor_list1.GetSize(), 6);
    EXPECT_EQ(tensor_list1.GetReservedSize(), 16);
    EXPECT_EQ(tensor_list1.AsTensor().ToFlatVector<float>(),
              std::vector<float>({1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                                  3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1,
                                  2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3}));
}

TEST_P(TensorListPermuteDevices, Clear) {
    core::Device device = GetParam();

    core::Tensor t0(std::vector<float>(2 * 3, 0), {2, 3}, core::Dtype::Float32,
                    device);
    std::vector<core::Tensor> tensors = {t0};

    core::TensorList tensor_list(tensors);
    tensor_list.Clear();
    EXPECT_EQ(tensor_list.GetSize(), 0);
    EXPECT_EQ(tensor_list.GetReservedSize(), 1);
}

}  // namespace tests
}  // namespace open3d
