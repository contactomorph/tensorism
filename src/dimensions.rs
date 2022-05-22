//! Abstractions for dimensions used in a tensor.

use rand::Rng;
use std::fmt::{Debug, Error, Formatter};
use std::marker::{Copy, PhantomData};

#[doc(hidden)]
pub fn generate_thumbprint() -> u16 {
    let mut rng = rand::thread_rng();
    rng.gen::<u16>()
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
            formatter.write_str("|")?;
            let b64 = base64::encode(tp.to_ne_bytes());
            formatter.write_str(b64.as_str())?;
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
macro_rules! new_dim {
    ($integer:expr) => {{
        #[allow(bad_style)]
        enum UnnamedTag {}
        lazy_static! {
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
