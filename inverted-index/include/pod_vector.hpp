#pragma once
// std::vector whose resize() leaves new elements uninitialized (default-init, not
// value-init). Large output arrays are then first touched by the parallel threads
// that fill them, instead of being zero-filled by one thread (a serial bottleneck).

#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace invidx {

template <class T, class A = std::allocator<T>>
struct default_init_allocator : A {
    using A::A;
    template <class U>
    struct rebind {
        using other = default_init_allocator<U, typename std::allocator_traits<A>::template rebind_alloc<U>>;
    };
    template <class U>
    void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>) {
        ::new (static_cast<void*>(p)) U;
    }
    template <class U, class... Args>
    void construct(U* p, Args&&... args) {
        std::allocator_traits<A>::construct(static_cast<A&>(*this), p, std::forward<Args>(args)...);
    }
};

template <class T>
using pod_vector = std::vector<T, default_init_allocator<T>>;

}  // namespace invidx
