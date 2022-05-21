//! Abstractions for dimensions used in a tensor.

use std::fmt::{Debug, Error, Formatter};
use std::marker::{Copy, PhantomData};
use std::sync::atomic::{AtomicU16, Ordering};

static COUNTER: AtomicU16 = AtomicU16::new(0xF091);

#[doc(hidden)]
pub fn generate_thumbprint() -> u16 {
    COUNTER
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
            Some(n.wrapping_mul(0x9CA3))
        })
        .unwrap()
}

/// An abstraction representing a static or dynamic information
/// about a dimension.
pub trait DimTag {
    fn get_thumbprint() -> u16;
}

pub struct StaticDimTag<const N: usize> {}

impl<const N: usize> DimTag for StaticDimTag<N> {
    fn get_thumbprint() -> u16 {
        0
    }
}

/// A type wrapping an usize value and representing
/// a dimension.
pub struct Dim<D: DimTag> {
    phantom: PhantomData<D>,
    v: usize,
}

impl<D: DimTag> Dim<D> {
    #[doc(hidden)]
    pub unsafe fn unsafe_new(v: usize) -> Self {
        Dim {
            phantom: PhantomData {},
            v,
        }
    }

    pub fn as_usize(&self) -> usize {
        self.v
    }
}

impl<D: DimTag> Debug for Dim<D> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), Error> {
        let tp = D::get_thumbprint();
        self.v.fmt(formatter)?;
        if tp != 0u16 {
            formatter.write_fmt(format_args!("|{:04x}", tp))?;
        }
        Ok(())
    }
}

impl<D: DimTag> Clone for Dim<D> {
    fn clone(&self) -> Self {
        Dim {
            phantom: PhantomData {},
            v: self.v,
        }
    }
}

impl<D: DimTag> Copy for Dim<D> {}

impl<TLhs: DimTag, TRhs: DimTag> PartialEq<Dim<TRhs>> for Dim<TLhs> {
    fn eq(&self, other: &Dim<TRhs>) -> bool {
        self.v == other.v
    }
}

impl<D: DimTag> Eq for Dim<D> {}

/// An alias representing a staticly known dimension
pub type StaticDim<const N: usize> = Dim<StaticDimTag<N>>;

#[macro_export]
macro_rules! new_dynamic_dim {
    ($integer:expr) => {{
        enum UnnamedTag {}
        lazy_static::lazy_static! {
            static ref THUMBPRINT: u16 = generate_thumbprint();
        }
        impl DimTag for UnnamedTag {
            fn get_thumbprint() -> u16 {
                *THUMBPRINT
            }
        }
        unsafe { Dim::<UnnamedTag>::unsafe_new($integer) }
    }};
}

pub fn new_static_dim<const N: usize>() -> Dim<StaticDimTag<N>> {
    Dim {
        phantom: PhantomData {},
        v: N,
    }
}

impl<D: DimTag> From<Dim<D>> for usize {
    fn from(dim: Dim<D>) -> Self {
        dim.v
    }
}
