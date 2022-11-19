//! Types for handling arrays with multiple indexes.
//!
//! Provides types for arrays with multiple indexes (up to 8 indexes) that are:
//! - convenient to manipulate thanks to [associated macros](https://docs.rs/tensorism-gen/latest/tensorism-gen/);
//! - safe to use as dimensions are encoded in the type system;
//! - minimising the number of runtime checks about validity of indexes.
//!
//! # Tensor types and dimensions
//!
//! Tensor types ([Tensor0](tensors::Tensor0), [Tensor1](tensors::Tensor1), [Tensor2](tensors::Tensor2),
//! [Tensor3](tensors::Tensor3), [Tensor4](tensors::Tensor4), [Tensor5](tensors::Tensor5), [Tensor6](tensors::Tensor6),
//! [Tensor7](tensors::Tensor7) and [Tensor8](tensors::Tensor8)) are generic types not only over the type of data they
//! contains but also over their multiple dimensions. This means dimensions are encoded in the type system.
//! ```
//! // A tensor of usizes of rank 3 with static dimensions 3, 2 and 5.
//! let tensor_a: Tensor3<StaticDimTag<3>, StaticDimTag<2>, StaticDimTag<5>, usize> =
//!   TensorBuilding::with_static::<3>()
//!     .with_static::<2>()
//!     .with_static::<5>()
//!     .define(|(i, j, k)| (3 * i + 2 * j + 5 * k) % 17);
//! // A tensor of strings of rank 1 with static dimensions 130.
//! let tensor_b: Tensor1<StaticDimTag<130>, String> =
//!   TensorBuilding::with_static::<130>()
//!     .prepare()
//!     .append_vec(&mut string_vector)
//!     .generate();
//! ```
//! This is not only true for compile-time dimensions. You can also create tensors some dimensions of which
//! are not known until runtime and still telling the compiler where these dimensions are equal, independently
//! of their actual numerical value.
//! ```
//! let n: usize = …;
//! let m: usize = …;
//! // A tensor of floats of rank 4 with dimensions n, 3, m and n.
//! // Its type is similar to `Tensor4<TagN, StaticDimTag<3>, TagM, TagN, f64>`.
//! let tensor_c: Tensor4<_, StaticDimTag<3>, _, _, f64> =
//!   TensorBuilding::with(new_dynamic_dim!(n))
//!     .with_static::<3>()
//!     .with(new_dynamic_dim!(m))
//!     .with_first()
//!     .define(|(i, j, k, l)| … as f64);
//!
//! // A tensor of booleans of rank 2 with dimensions m and n.
//! // Its type is similar to `Tensor4<TagM, TagN, bool>`.
//! let tensor_d: Tensor2<_, _, bool> =
//!   TensorBuilding::with(tensor_c.shape().2)
//!     .with(tensor_c.shape().0)
//!     .define(|(i, j)| … as bool);
//! ```
//! 
//! # Manipulating tensors
//! 
//! Sibling crate [tensorism-gen](https://docs.rs/tensorism-gen/latest/tensorism-gen/) contains procedural
//! macros that greatly simplify the effective manipulation of tensors via a domain specific language. Please
//! refer to this crate for additional documentation.
#![cfg(not(doctest))]
extern crate lazy_static;
pub mod building;
pub mod dimensions;
pub mod tensors;

/// Generate a new dimension at runtime.
/// 
/// The parameter must be of type [usize]. This macro returns a value of type
/// [Dim<_>](dimensions::Dim) that uses a newly generated unnamed tag.
/// If the dimension is a compile-time constant, prefer [dimensions::new_static_dim].
#[macro_export]
macro_rules! new_dynamic_dim {
    ($integer:expr) => {{
        #[derive(PartialEq, Clone, Copy)]
        enum DynDimTag {}
        lazy_static::lazy_static! {
            static ref THUMBPRINT: u16 = ::tensorism::dimensions::generate_thumbprint();
        }
        impl ::tensorism::dimensions::DimTag for DynDimTag {
            fn get_thumbprint() -> Option<u16> {
                Some(*THUMBPRINT)
            }
        }
        unsafe { ::tensorism::dimensions::Dim::<DynDimTag>::unsafe_new($integer) }
    }};
}

/// Transform a [Vec] into a [Tensor1](tensors::Tensor1).
/// 
/// Expects a [Vec\<T\>](Vec) parameter and returns a [Tensor1\<_, T\>](tensors::Tensor1).
/// A newly generated unnamed tag is used to represent the length of the input.
#[macro_export]
macro_rules! from_vec_to_tensor1 {
    ($values:expr) => {{
        #[derive(PartialEq, Clone, Copy)]
        enum DynDimTag {}
        lazy_static::lazy_static! {
            static ref THUMBPRINT: u16 = ::tensorism::dimensions::generate_thumbprint();
        }
        impl ::tensorism::dimensions::DimTag for DynDimTag {
            fn get_thumbprint() -> u16 {
                *THUMBPRINT
            }
        }
        unsafe { ::tensorism::building::from_vec_to_tensor1_unchecked::<DynDimTag, _>($values) }
    }};
}
