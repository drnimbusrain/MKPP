#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <type_traits>

namespace mkpp::generated::saprc99::detail {

/**
 * @brief Layout-aware, non-owning accessor over a rank-1 or rank-2 buffer.
 *
 * @tparam T     Element type, possibly const-qualified.
 * @tparam Rank  Buffer rank; only 1 and 2 are supported.
 *
 * Element (i) resolves to data_[i * stride(0)] and (i, j) to
 * data_[i * stride(0) + j * stride(1)], i.e. exactly the view's own mapping,
 * whatever layout the caller bound.
 */
template <class T, unsigned Rank>
struct StrideView {
  // A rank outside {1, 2} would silently drop a stride and reintroduce the
  // flat-index defect this accessor exists to prevent.
  static_assert(Rank == 1 || Rank == 2, "StrideView supports rank 1 and rank 2 buffers only");
};

/// Rank-1 accessor: one strided index (a cell's species, or a species' cells).
template <class T>
class StrideView<T, 1> {
 public:
  using value_type = T;
  using pointer_type = T*;
  static constexpr unsigned rank = 1;

  KOKKOS_DEFAULTED_FUNCTION StrideView() = default;

  KOKKOS_INLINE_FUNCTION StrideView(T* data, std::size_t stride0) : data_(data), stride0_(stride0) {}

  /// Relaxes a mutable accessor to a const one; never the reverse.
  template <class U>
    requires(!std::is_same_v<StrideView, StrideView<U, 1>> && std::is_same_v<T, const U>)
  KOKKOS_INLINE_FUNCTION StrideView(const StrideView<U, 1>& other)
      : data_(other.data()), stride0_(other.stride(0)) {}

  KOKKOS_INLINE_FUNCTION value_type& operator()(std::size_t i) const { return data_[i * stride0_]; }

  KOKKOS_INLINE_FUNCTION pointer_type data() const { return data_; }
  KOKKOS_INLINE_FUNCTION std::size_t stride(unsigned r) const { return r == 0 ? stride0_ : 1u; }

 private:
  pointer_type data_ = nullptr;
  std::size_t stride0_ = 1u;
};

/// Rank-2 accessor: row/column strides kept distinct (a cell's Jacobian block).
template <class T>
class StrideView<T, 2> {
 public:
  using value_type = T;
  using pointer_type = T*;
  static constexpr unsigned rank = 2;

  KOKKOS_DEFAULTED_FUNCTION StrideView() = default;

  KOKKOS_INLINE_FUNCTION StrideView(T* data, std::size_t stride0, std::size_t stride1)
      : data_(data), stride0_(stride0), stride1_(stride1) {}

  /// Relaxes a mutable accessor to a const one; never the reverse.
  template <class U>
    requires(!std::is_same_v<StrideView, StrideView<U, 2>> && std::is_same_v<T, const U>)
  KOKKOS_INLINE_FUNCTION StrideView(const StrideView<U, 2>& other)
      : data_(other.data()), stride0_(other.stride(0)), stride1_(other.stride(1)) {}

  KOKKOS_INLINE_FUNCTION value_type& operator()(std::size_t i, std::size_t j) const {
    return data_[i * stride0_ + j * stride1_];
  }

  KOKKOS_INLINE_FUNCTION static StrideView compact(T* data) {
    StrideView result(data, 0, 0);
    return result;
  }

  KOKKOS_INLINE_FUNCTION value_type& at_slot(std::size_t slot, std::size_t i, std::size_t j) const {
    return stride0_ == 0 && stride1_ == 0 ? data_[slot] : (*this)(i, j);
  }

  KOKKOS_INLINE_FUNCTION pointer_type data() const { return data_; }
  KOKKOS_INLINE_FUNCTION std::size_t stride(unsigned r) const { return r == 0 ? stride0_ : (r == 1 ? stride1_ : 1u); }

 private:
  pointer_type data_ = nullptr;
  std::size_t stride0_ = 1u;
  std::size_t stride1_ = 1u;
};

/**
 * @brief Builds the accessor that reproduces a Kokkos View's own layout.
 *
 * Reads the view's data pointer and per-dimension strides, so a LayoutLeft
 * cell-row, a LayoutRight cell-row and a LayoutStride subview all map to the
 * same logical elements.
 */
template <class ViewType>
KOKKOS_INLINE_FUNCTION auto make_stride_view(const ViewType& view) {
  // Only layouts Kokkos can describe with strides are addressable this way;
  // anything else must fail at compile time rather than mis-index at runtime.
  static_assert(std::is_same_v<typename ViewType::array_layout, Kokkos::LayoutLeft> ||
                    std::is_same_v<typename ViewType::array_layout, Kokkos::LayoutRight> ||
                    std::is_same_v<typename ViewType::array_layout, Kokkos::LayoutStride>,
                "make_stride_view: caller buffer must use a stride-describable Kokkos layout");
  static_assert(ViewType::rank == 1 || ViewType::rank == 2,
                "make_stride_view: caller buffers must be rank 1 or rank 2");

  using pointer_type = decltype(view.data());
  using element_type = std::remove_pointer_t<pointer_type>;
  if constexpr (ViewType::rank == 1) {
    return StrideView<element_type, 1>(view.data(), view.stride(0));
  } else {
    return StrideView<element_type, 2>(view.data(), view.stride(0), view.stride(1));
  }
}

/**
 * @brief Builds the accessor over a contiguous stack scratch array.
 *
 * The integrator's local J_values/F_values buffers are row-major by
 * construction, so the strides are the compile-time species count and one.
 * Indexing through the accessor therefore reproduces the former flat
 * arr[i*N + j] writes exactly, while keeping one pathway for every buffer.
 */
template <class T, unsigned Rank>
KOKKOS_INLINE_FUNCTION StrideView<T, Rank> make_stride_view(T* data, std::size_t stride0, std::size_t stride1 = 1u) {
  static_assert(Rank == 1 || Rank == 2, "make_stride_view: supports rank 1 and rank 2 buffers only");
  if constexpr (Rank == 1) {
    return StrideView<T, 1>(data, stride0);
  } else {
    return StrideView<T, 2>(data, stride0, stride1);
  }
}

}  // namespace mkpp::generated::saprc99::detail
