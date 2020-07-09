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
#include "open3d/core/SizeVector.h"

namespace open3d {
namespace core {

TensorList TensorList::FromTensor(const Tensor& tensor, bool inplace) {
    SizeVector shape = tensor.GetShape();
    if (shape.size() == 0) {
        utility::LogError("Tensor should at least have one dimension.");
    }
    SizeVector element_shape =
            SizeVector(std::next(shape.begin()), shape.end());
    int64_t size = shape[0];

    if (inplace) {
        if (!tensor.IsContiguous()) {
            utility::LogError(
                    "Tensor must be contiguous for inplace TensorList "
                    "construction.");
        }
        return TensorList(element_shape, size, size, tensor,
                          /*is_resizable=*/false);
    } else {
        int64_t reserved_size = TensorList::ComputeReserveSize(size);
        Tensor internal_tensor = Tensor::Empty(
                shape_util::ConcatShapes({reserved_size}, element_shape),
                tensor.GetDtype(), tensor.GetDevice());
        internal_tensor.Slice(0, 0, size) = tensor;
        return TensorList(element_shape, size, reserved_size, internal_tensor,
                          /*is_resizable=*/true);
    }
}

void TensorList::CopyFrom(const TensorList& other) {
    ShallowCopyFrom(other);
    internal_tensor_ = other.AsTensor().Copy();
}

void TensorList::ShallowCopyFrom(const TensorList& other) {
    // Copy assignment operator is performing shallow copy.
    *this = other;
}

Tensor TensorList::AsTensor() const {
    return internal_tensor_.Slice(/*dim=*/0, 0, size_);
}

void TensorList::Resize(int64_t n) {
    if (!is_resizable_) {
        utility::LogError(
                "TensorList is not resizable. Typically this TensorList is "
                "created with shared memory from a Tensor.");
    }

    // Increase internal tensor size.
    MaybeExpandTensor(n);

    // Initialize with 0.
    if (n > size_) {
        internal_tensor_.Slice(/*dim=*/0, size_, n).Fill(0);
    }
    size_ = n;
}

void TensorList::PushBack(const Tensor& tensor) {
    if (!is_resizable_) {
        utility::LogError(
                "TensorList is not resizable. Typically this TensorList is "
                "created with shared memory from a Tensor.");
    }
    if (element_shape_ != tensor.GetShape()) {
        utility::LogError(
                "TensorList has element shape {}, but tensor has shape {}.",
                element_shape_, tensor.GetShape());
    }
    if (GetDtype() != tensor.GetDtype()) {
        utility::LogError("TensorList has dtype {}, but tensor has shape {}.",
                          DtypeUtil::ToString(GetDtype()),
                          DtypeUtil::ToString(tensor.GetDtype()));
    }
    if (GetDevice() != tensor.GetDevice()) {
        utility::LogError("TensorList has device {}, but tensor has shape {}.",
                          GetDevice().ToString(),
                          tensor.GetDevice().ToString());
    }
    MaybeExpandTensor(size_ + 1);
    internal_tensor_[size_] = tensor;
    ++size_;
}

TensorList TensorList::Concatenate(const TensorList& a, const TensorList& b) {
    TensorList result(a);
    result.Extend(b);
    return result;
}

void TensorList::Extend(const TensorList& other) {
    if (!is_resizable_) {
        utility::LogError(
                "TensorList is not resizable. Typically this TensorList is "
                "created with shared memory from a Tensor.");
    }

    // Check consistency
    if (element_shape_ != other.GetElementShape()) {
        utility::LogError("TensorList shapes {} and {} are inconsistent.",
                          element_shape_, other.GetElementShape());
    }

    if (GetDevice() != other.GetDevice()) {
        utility::LogError("TensorList device {} and {} are inconsistent.",
                          GetDevice().ToString(), other.GetDevice().ToString());
    }

    if (GetDtype() != other.GetDtype()) {
        utility::LogError("TensorList dtype {} and {} are inconsistent.",
                          DtypeUtil::ToString(GetDtype()),
                          DtypeUtil::ToString(other.GetDtype()));
    }

    // Ignore empty TensorList
    if (other.GetSize() == 0) {
        return;
    }

    // Shallow copy by default
    TensorList extension = other;

    // Make a deep copy to avoid corrupting duplicate data
    if (GetInternalTensor().GetDataPtr() ==
        other.GetInternalTensor().GetDataPtr()) {
        extension = TensorList(*this);
    }

    MaybeExpandTensor(size_ + extension.GetSize());
    internal_tensor_.Slice(/*dim=*/0, size_, size_ + extension.GetSize()) =
            extension.AsTensor();
    size_ = size_ + extension.GetSize();
}

Tensor TensorList::operator[](int64_t index) const {
    index = shape_util::WrapDim(
            index, size_);  // WrapDim asserts index is within range.
    return internal_tensor_[index];
}

void TensorList::Clear() {
    *this = TensorList(element_shape_, GetDtype(), GetDevice());
}

// Protected
void TensorList::MaybeExpandTensor(int64_t new_size) {
    int64_t new_reserved_size = ComputeReserveSize(new_size);
    if (new_reserved_size <= reserved_size_) {
        return;
    }

    SizeVector new_expanded_shape =
            shape_util::ConcatShapes({new_reserved_size}, element_shape_);
    Tensor new_internal_tensor =
            Tensor(new_expanded_shape, GetDtype(), GetDevice());

    // Copy data
    new_internal_tensor.Slice(/*dim=*/0, 0, size_) =
            internal_tensor_.Slice(/*dim=*/0, 0, size_);
    internal_tensor_ = new_internal_tensor;
    reserved_size_ = new_reserved_size;
}

int64_t TensorList::ComputeReserveSize(int64_t n) {
    if (n < 0) {
        utility::LogError("Negative tensor list size {} is not supported.", n);
    }

    int64_t base = 1;
    if (n > (base << 61)) {
        utility::LogError("Too large tensor list size {} is not supported.", n);
    }

    for (int i = 63; i >= 0; --i) {
        // First nnz bit
        if (((base << i) & n) > 0) {
            if (n == (base << i)) {
                // Power of 2: 2 * n. For instance, 8 tensors will be
                // reserved for size=4
                return (base << (i + 1));
            } else {
                // Non-power of 2: ceil(log(2)) * 2. For instance, 16
                // tensors will be reserved for size=5
                return (base << (i + 2));
            }
        }
    }

    // No nnz bit: by default reserve 1 element.
    return 1;
}

std::string TensorList::ToString() const {
    std::ostringstream rc;
    rc << fmt::format("\nTensorList[size={}, shape={}, {}, {}]", size_,
                      element_shape_.ToString(),
                      DtypeUtil::ToString(GetDtype()), GetDevice().ToString());
    return rc.str();
}
}  // namespace core
}  // namespace open3d
