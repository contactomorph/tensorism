use crate::dimensions::{new_static_dim, Dim, DimTag, StaticDimTag};
use std::fmt::{Debug, Error, Formatter};

pub trait Shape {
    /// The number of dimensions, also corresponding
    /// to the number of indexes.
    const RANK: u16;
    /// The total number of coordinates
    fn count(&self) -> usize;
}

#[derive(Clone, Copy)]
pub struct Tensor1Shape<T: DimTag> {
    pub(crate) d: Dim<T>,
    size: usize,
}

#[derive(Clone, Copy)]
pub struct Tensor2Shape<T1: DimTag, T2: DimTag> {
    pub(crate) d1: Dim<T1>,
    pub(crate) d2: Dim<T2>,
    size: usize,
}

#[derive(Clone, Copy)]
pub struct Tensor3Shape<T1: DimTag, T2: DimTag, T3: DimTag> {
    pub(crate) d1: Dim<T1>,
    pub(crate) d2: Dim<T2>,
    pub(crate) d3: Dim<T3>,
    size: usize,
}

impl<T: DimTag> Shape for Tensor1Shape<T> {
    const RANK: u16 = 1;

    fn count(&self) -> usize {
        self.size
    }
}

impl<T1: DimTag, T2: DimTag> Shape for Tensor2Shape<T1, T2> {
    const RANK: u16 = 2;

    fn count(&self) -> usize {
        self.size
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag> Shape for Tensor3Shape<T1, T2, T3> {
    const RANK: u16 = 3;

    fn count(&self) -> usize {
        self.size
    }
}

impl<L: DimTag, R: DimTag> PartialEq<Tensor1Shape<R>> for Tensor1Shape<L> {
    fn eq(&self, other: &Tensor1Shape<R>) -> bool {
        self.d == other.d
    }
}

impl<T: DimTag> Eq for Tensor1Shape<T> {}

impl<L1: DimTag, L2: DimTag, R1: DimTag, R2: DimTag> PartialEq<Tensor2Shape<R1, R2>>
    for Tensor2Shape<L1, L2>
{
    fn eq(&self, other: &Tensor2Shape<R1, R2>) -> bool {
        self.d1 == other.d1 && self.d2 == other.d2
    }
}

impl<T1: DimTag, T2: DimTag> Eq for Tensor2Shape<T1, T2> {}

impl<L1: DimTag, L2: DimTag, L3: DimTag, R1: DimTag, R2: DimTag, R3: DimTag>
    PartialEq<Tensor3Shape<R1, R2, R3>> for Tensor3Shape<L1, L2, L3>
{
    fn eq(&self, other: &Tensor3Shape<R1, R2, R3>) -> bool {
        self.d1 == other.d1 && self.d2 == other.d2 && self.d3 == other.d3
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag> Eq for Tensor3Shape<T1, T2, T3> {}

impl<T: DimTag> Debug for Tensor1Shape<T> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), Error> {
        formatter.write_str("\u{3008}")?;
        self.d.fmt(formatter)?;
        formatter.write_str("\u{3009}")
    }
}

impl<T1: DimTag, T2: DimTag> Debug for Tensor2Shape<T1, T2> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), Error> {
        formatter.write_str("\u{3008}")?;
        self.d1.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d2.fmt(formatter)?;
        formatter.write_str("\u{3009}")
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag> Debug for Tensor3Shape<T1, T2, T3> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), Error> {
        formatter.write_str("\u{3008}")?;
        self.d1.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d2.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d3.fmt(formatter)?;
        formatter.write_str("\u{3009}")
    }
}

pub enum ShapeBuilder {}

impl ShapeBuilder {
    pub fn with_static<const N: usize>() -> Tensor1Shape<StaticDimTag<N>> {
        Tensor1Shape {
            d: new_static_dim::<N>(),
            size: N,
        }
    }
    pub fn with<T: DimTag>(dimension: Dim<T>) -> Tensor1Shape<T> {
        Tensor1Shape {
            d: dimension,
            size: dimension.as_usize(),
        }
    }
}

impl<T: DimTag> Tensor1Shape<T> {
    pub fn with_static<const N: usize>(&self) -> Tensor2Shape<T, StaticDimTag<N>> {
        Tensor2Shape {
            d1: self.d,
            d2: new_static_dim::<N>(),
            size: self.size.checked_mul(N).expect("Tensor size is too big"),
        }
    }
    pub fn with<T2: DimTag>(&self, dimension: Dim<T2>) -> Tensor2Shape<T, T2> {
        Tensor2Shape {
            d1: self.d,
            d2: dimension,
            size: self
                .size
                .checked_mul(dimension.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_first(&self) -> Tensor2Shape<T, T> {
        Tensor2Shape {
            d1: self.d,
            d2: self.d,
            size: self
                .size
                .checked_mul(self.size)
                .expect("Tensor size is too big"),
        }
    }
}

impl<T1: DimTag, T2: DimTag> Tensor2Shape<T1, T2> {
    pub fn with_static<const N: usize>(&self) -> Tensor3Shape<T1, T2, StaticDimTag<N>> {
        Tensor3Shape {
            d1: self.d1,
            d2: self.d2,
            d3: new_static_dim::<N>(),
            size: self.size.checked_mul(N).expect("Tensor size is too big"),
        }
    }
    pub fn with<T3: DimTag>(&self, dimension: Dim<T3>) -> Tensor3Shape<T1, T2, T3> {
        Tensor3Shape {
            d1: self.d1,
            d2: self.d2,
            d3: dimension,
            size: self
                .size
                .checked_mul(dimension.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_first(&self) -> Tensor3Shape<T1, T2, T1> {
        Tensor3Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d1,
            size: self
                .size
                .checked_mul(self.d1.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_second(&self) -> Tensor3Shape<T1, T2, T2> {
        Tensor3Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d2,
            size: self
                .size
                .checked_mul(self.d2.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn switch_12(&self) -> Tensor2Shape<T2, T1> {
        Tensor2Shape {
            d1: self.d2,
            d2: self.d1,
            size: self.size,
        }
    }
}
