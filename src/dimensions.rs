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
/// about a dimension.
pub trait DimTag: Eq + Copy {
    fn get_thumbprint() -> Option<u16>;
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub struct StaticDimTag<const N: usize> {}

impl<const N: usize> DimTag for StaticDimTag<N> {
    fn get_thumbprint() -> Option<u16> {
        None
    }
}

/// A type wrapping an usize value and representing
/// a dimension.
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

    pub fn as_usize(&self) -> usize {
        self.v
    }

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

impl<T: DimTag> Eq for Dim<T> {}

/// An alias representing a staticly known dimension
pub type StaticDim<const N: usize> = Dim<StaticDimTag<N>>;

#[macro_export]
macro_rules! new_dynamic_dim {
    ($integer:expr) => {{
        #[derive(PartialEq, Eq, Debug, Clone, Copy)]
        enum DynDimTag {}
        lazy_static::lazy_static! {
            static ref THUMBPRINT: u16 = generate_thumbprint();
        }
        impl DimTag for DynDimTag {
            fn get_thumbprint() -> Option<u16> {
                Some(*THUMBPRINT)
            }
        }
        unsafe { Dim::<DynDimTag>::unsafe_new($integer) }
    }};
}

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
