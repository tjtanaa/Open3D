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

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "open3d/core/Blob.h"
#include "open3d/core/Device.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/TensorKey.h"
namespace open3d {
namespace core {

/// A TensorList is a list of Tensors of the same shape, similar to
/// std::vector<Tensor>. Internally, a TensorList stores the Tensors in one
/// bigger internal tensor, where the begin dimension of the internal tensor is
/// extendable.
///
/// Examples:
/// - A 3D point cloud with N points:
///   - element_shape        : (3,)
///   - reserved_size        : M, where M >= N
///   - internal_tensor.shape: (M, 3)
/// - Sparse voxel grid of N voxels:
///   - element_shape        : (8, 8, 8)
///   - reserved_size        : M, where M >= N
///   - internal_tensor.shape: (M, 8, 8, 8)
class TensorList {
public:
    TensorList() = delete;

    /// Constructs an emtpy TensorList.
    ///
    /// \param element_shape Shape of the contained tensors, e.g. (3,).
    /// \param dtype Data type of the contained tensors. e.g. Dtype::Float32.
    /// \param device Device of the contained tensors. e.g. Device("CPU:0").
    /// \param size Number of initial tensors, similar to std::vector<T>::size.
    TensorList(const SizeVector& element_shape,
               Dtype dtype,
               const Device& device = Device("CPU:0"),
               const int64_t& size = 0)
        : element_shape_(element_shape),
          size_(size),
          reserved_size_(ReserveSize(size)),
          internal_tensor_(ExpandFrontDim(element_shape_, reserved_size_),
                           dtype,
                           device) {}

    /// Constructs a TensorList from a vector of Tensors.
    ///
    /// \param tensors A vector of tensors. The tensors must have common shape,
    /// dtype and device.
    TensorList(const std::vector<Tensor>& tensors)
        : TensorList(tensors.begin(), tensors.end()) {}

    /// Constructs a TensorList from a list of Tensors.
    ///
    /// \param tensors A list of tensors. The tensors must have common shape,
    /// dtype and device.
    TensorList(const std::initializer_list<Tensor>& tensors)
        : TensorList(tensors.begin(), tensors.end()) {}

    /// Constructs a TensorList from Tensor iterator. The tensors must have
    /// common shape, dtype and device.
    ///
    /// \param begin Begin iterator.
    /// \param end End iterator.
    template <class InputIterator>
    TensorList(InputIterator begin, InputIterator end) {
        int64_t size = std::distance(begin, end);
        if (size == 0) {
            utility::LogError(
                    "Empty input tensors cannot initialize a TensorList.");
        }

        // Set size_ and reserved_size_.
        size_ = size;
        reserved_size_ = ReserveSize(size_);

        // Check shape consistency and set element_shape_.
        element_shape_ = begin->GetShape();
        std::for_each(begin, end, [&](const Tensor& tensor) -> void {
            if (tensor.GetShape() != element_shape_) {
                utility::LogError(
                        "Tensors must have the same shape {}, but got {}.",
                        element_shape_, tensor.GetShape());
            }
        });

        // Check dtype consistency.
        Dtype dtype = begin->GetDtype();
        std::for_each(begin, end, [&](const Tensor& tensor) -> void {
            if (tensor.GetDtype() != dtype) {
                utility::LogError(
                        "Tensors must have the same dtype {}, but got {}.",
                        DtypeUtil::ToString(dtype),
                        DtypeUtil::ToString(tensor.GetDtype()));
            }
        });

        // Check device consistency.
        Device device = begin->GetDevice();
        std::for_each(begin, end, [&](const Tensor& tensor) -> void {
            if (tensor.GetDevice() != device) {
                utility::LogError(
                        "Tensors must have the same device {}, but got {}.",
                        device.ToString(), tensor.GetDevice().ToString());
            }
        });

        // Construct internal tensor.
        SizeVector expanded_shape =
                ExpandFrontDim(element_shape_, reserved_size_);
        internal_tensor_ = Tensor(expanded_shape, dtype, device);
        size_t i = 0;
        for (auto iter = begin; iter != end; ++iter, ++i) {
            internal_tensor_[i] = *iter;
        }
    }

    /// Constructor from a raw internal tensor.
    /// The inverse of AsTensor().
    ///
    /// \param internal_tensor raw tensor
    /// \param inplace:
    /// - If true (default), reuse the raw internal tensor. The input tensor
    /// must be contiguous.
    /// - If false, create a new contiguous internal tensor with precomputed
    /// reserved size.
    TensorList(const Tensor& internal_tensor, bool inplace = true);

    /// Factory constructor from a raw tensor
    static TensorList FromTensor(const Tensor& tensor, bool inplace = false);

    /// Copy constructor from a tensor list.
    /// Create a new tensor list with copy of data.
    TensorList(const TensorList& other);

    /// Deep copy
    void CopyFrom(const TensorList& other);

    /// TensorList assignment lvalue = lvalue, e.g.
    /// `tensorlist_a = tensorlist_b`,
    /// resulting in a shallow copy.
    /// We don't redirect Slice operation to tensors, so right value assignment
    /// is not explicitly supported.
    TensorList& operator=(const TensorList& other) &;

    /// Shallow copy
    void ShallowCopyFrom(const TensorList& other);

    /// Return the reference of the contained valid tensors with shared memory.
    Tensor AsTensor() const;

    /// Resize an existing tensor list.
    /// If the size increases, the increased part will be assigned 0.
    /// If the size decreases, the decreased part's value will be undefined.
    void Resize(int64_t n);

    /// Push back the copy of a tensor to the list.
    /// The tensor must broadcastable to the TensorList's element_shape.
    /// The tensor must be on the same device and have the same dtype.
    void PushBack(const Tensor& tensor);

    /// Concatenate two TensorLists.
    /// Return a new TensorList with data copied.
    /// Two TensorLists must have the same element_shape, type, and device.
    static TensorList Concatenate(const TensorList& a, const TensorList& b);

    /// Concatenate two TensorLists.
    TensorList operator+(const TensorList& other) const {
        return Concatenate(*this, other);
    }

    /// Extend the current TensorList with another TensorList appended to the
    /// end. The data is copied. The two TensorLists must have the same
    /// element_shape, dtype, and device.
    void Extend(const TensorList& other);

    TensorList& operator+=(const TensorList& other) {
        Extend(other);
        return *this;
    }

    /// Extract the i-th Tensor along the begin axis, returning a new view.
    /// For advanced indexing like Slice, use tensorlist.AsTensor().Slice().
    Tensor operator[](int64_t index) const;

    /// Clear the tensor list by discarding all data and creating a empty one.
    void Clear();

    std::string ToString() const;

    SizeVector GetElementShape() const { return element_shape_; }

    Device GetDevice() const { return internal_tensor_.GetDevice(); }

    Dtype GetDtype() const { return internal_tensor_.GetDtype(); }

    int64_t GetSize() const { return size_; }

    int64_t GetReservedSize() const { return reserved_size_; }

    const Tensor& GetInternalTensor() const { return internal_tensor_; }

protected:
    /// Expand the size of the internal tensor.
    void ExpandTensor(int64_t new_reserved_size);

    /// Expand the element_shape in the begin indexing dimension.
    /// e.g. (8, 8, 8) -> (1, 8, 8, 8)
    static SizeVector ExpandFrontDim(const SizeVector& element_shape,
                                     int64_t new_dim_size = 1);

    /// Compute the reserved size for the desired number of tensors
    /// with reserved_size_ = (1 << (ceil(log2(size_)) + 1)).
    int64_t ReserveSize(int64_t n);

protected:
    /// The shape for each element tensor in the TensorList.
    SizeVector element_shape_;

    /// Number of active (valid) elements in TensorList.
    /// The internal_tensor_ has shape (reserved_size_, *shape_), but only the
    /// front (size_, *shape_) is active.
    int64_t size_ = 0;

    /// Maximum number of elements in TensorList.
    ///
    /// The internal_tensor_'s shape is (reserved_size_, *element_shape_). In
    /// general, reserved_size_ >= (1 << (ceil(log2(size_)) + 1)) as
    /// conventionally done in std::vector.
    ///
    /// Examples: (size_, reserved_size_) = (3, 8), (4, 8), (5, 16).
    int64_t reserved_size_ = 0;

    /// The internal tensor for data storage.
    Tensor internal_tensor_;
};
}  // namespace core
}  // namespace open3d
