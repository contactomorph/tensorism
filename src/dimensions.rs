//! Abstractions for dimensions used in a tensor.

use std::fmt::{Debug, Error, Formatter};
use std::marker::{Copy, PhantomData};
use std::sync::atomic::{AtomicU16, Ordering};

static COUNTER: AtomicU16 = AtomicU16::new(0xF091);

#[doc(hidden)]
pub fn generate_thumbprint() -> u16 {
    COUNTER
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
            Some(n.wrapping_mul(0x9CA3) | 0x34E1u16)
        })
        .unwrap()
}

/// An abstraction representing a static or dynamic information
/// about a dimension. Types implementing DimTag do not need to
/// be instanciable as their main purpose is to represent dimension
/// in the type system.
pub trait DimTag: PartialEq + Copy {
    /// Returns a u16 used to name a dynamic dimension or None
    /// if the dimension is statically known. This is mainly used
    /// for displaying debugging information.
    fn get_thumbprint() -> Option<u16>;
}

/// A tag representing a dimension known at compile time.
#[derive(PartialEq, Clone, Copy)]
pub enum StaticDimTag<const N: usize> {}

impl<const N: usize> DimTag for StaticDimTag<N> {
    fn get_thumbprint() -> Option<u16> {
        None
    }
}

/// A type wrapping a usize value and representing
/// a dimension. All safe construction methods of instances for
/// this type makes sure that two instances with the same tag
/// necessarily represent the same dimension.
#[derive(Clone, Copy)]
pub struct Dim<T: DimTag> {
    phantom: PhantomData<T>,
    v: usize,
}

impl<T: DimTag> Dim<T> {
    #[doc(hidden)]
    pub unsafe fn unsafe_new(v: usize) -> Self {
        Dim {
            phantom: PhantomData {},
            v,
        }
    }

    /// Return the dimension as a usize.
    pub fn as_usize(&self) -> usize {
        self.v
    }

    /// Returns the thumbprint of the T. As for tags
    /// this value is mainly used
    /// for displaying debugging information.
    pub fn get_thumbprint(&self) -> Option<u16> {
        T::get_thumbprint()
    }
}

impl<T: DimTag> Debug for Dim<T> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), Error> {
        let tp = T::get_thumbprint();
        self.v.fmt(formatter)?;
        if let Some(tp) = tp {
            formatter.write_fmt(format_args!("\u{2022}{:04x}", tp))?;
        }
        Ok(())
    }
}

impl<TLhs: DimTag, TRhs: DimTag> PartialEq<Dim<TRhs>> for Dim<TLhs> {
    fn eq(&self, other: &Dim<TRhs>) -> bool {
        self.v == other.v
    }
}

#[doc(hidden)]
pub fn identical<T: DimTag>(_d1: Dim<T>, _d2: Dim<T>) {}

impl<T: DimTag> Eq for Dim<T> {}

/// An alias representing a staticly known dimension
pub type StaticDim<const N: usize> = Dim<StaticDimTag<N>>;

/// Generate a new dimension wrapping the constant size N.
pub fn new_static_dim<const N: usize>() -> Dim<StaticDimTag<N>> {
    Dim {
        phantom: PhantomData {},
        v: N,
    }
}

impl<T: DimTag> From<Dim<T>> for usize {
    fn from(dim: Dim<T>) -> Self {
        dim.v
    }
}
