use crate::dimensions::*;
use crate::tensors::*;
use std::fmt::{Debug, Error, Formatter};
use std::marker::{Copy, PhantomData};

pub trait Shape {
    /// The number of dimensions, also corresponding
    /// to the number of indexes.
    const RANK: u16;
    /// The total number of coordinates
    fn count(&self) -> usize;
}

#[derive(Clone, Copy, Eq)]
pub struct Tensor1Shape<T: DimTag> {
    pub(crate) d: Dim<T>,
    size: usize,
}

#[derive(Clone, Copy, Eq)]
pub struct Tensor2Shape<T1: DimTag, T2: DimTag> {
    pub(crate) d1: Dim<T1>,
    pub(crate) d2: Dim<T2>,
    size: usize,
}

#[derive(Clone, Copy, Eq)]
pub struct Tensor3Shape<T1: DimTag, T2: DimTag, T3: DimTag> {
    pub(crate) d1: Dim<T1>,
    pub(crate) d2: Dim<T2>,
    pub(crate) d3: Dim<T3>,
    size: usize,
}

#[derive(Clone, Copy, Eq)]
pub struct Tensor4Shape<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag> {
    pub(crate) d1: Dim<T1>,
    pub(crate) d2: Dim<T2>,
    pub(crate) d3: Dim<T3>,
    pub(crate) d4: Dim<T4>,
    size: usize,
}

#[derive(Clone, Copy, Eq)]
pub struct Tensor5Shape<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag, T5: DimTag> {
    pub(crate) d1: Dim<T1>,
    pub(crate) d2: Dim<T2>,
    pub(crate) d3: Dim<T3>,
    pub(crate) d4: Dim<T4>,
    pub(crate) d5: Dim<T5>,
    size: usize,
}

#[derive(Clone, Copy, Eq)]
pub struct Tensor6Shape<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag, T5: DimTag, T6: DimTag> {
    pub(crate) d1: Dim<T1>,
    pub(crate) d2: Dim<T2>,
    pub(crate) d3: Dim<T3>,
    pub(crate) d4: Dim<T4>,
    pub(crate) d5: Dim<T5>,
    pub(crate) d6: Dim<T6>,
    size: usize,
}

#[derive(Clone, Copy)]
pub struct Tensor7Shape<
    T1: DimTag,
    T2: DimTag,
    T3: DimTag,
    T4: DimTag,
    T5: DimTag,
    T6: DimTag,
    T7: DimTag,
> {
    pub(crate) d1: Dim<T1>,
    pub(crate) d2: Dim<T2>,
    pub(crate) d3: Dim<T3>,
    pub(crate) d4: Dim<T4>,
    pub(crate) d5: Dim<T5>,
    pub(crate) d6: Dim<T6>,
    pub(crate) d7: Dim<T7>,
    size: usize,
}

#[derive(Clone, Copy)]
pub struct Tensor8Shape<
    T1: DimTag,
    T2: DimTag,
    T3: DimTag,
    T4: DimTag,
    T5: DimTag,
    T6: DimTag,
    T7: DimTag,
    T8: DimTag,
> {
    pub(crate) d1: Dim<T1>,
    pub(crate) d2: Dim<T2>,
    pub(crate) d3: Dim<T3>,
    pub(crate) d4: Dim<T4>,
    pub(crate) d5: Dim<T5>,
    pub(crate) d6: Dim<T6>,
    pub(crate) d7: Dim<T7>,
    pub(crate) d8: Dim<T8>,
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

impl<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag> Shape for Tensor4Shape<T1, T2, T3, T4> {
    const RANK: u16 = 4;

    fn count(&self) -> usize {
        self.size
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag, T5: DimTag> Shape
    for Tensor5Shape<T1, T2, T3, T4, T5>
{
    const RANK: u16 = 5;

    fn count(&self) -> usize {
        self.size
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag, T5: DimTag, T6: DimTag> Shape
    for Tensor6Shape<T1, T2, T3, T4, T5, T6>
{
    const RANK: u16 = 6;

    fn count(&self) -> usize {
        self.size
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag, T5: DimTag, T6: DimTag, T7: DimTag> Shape
    for Tensor7Shape<T1, T2, T3, T4, T5, T6, T7>
{
    const RANK: u16 = 7;

    fn count(&self) -> usize {
        self.size
    }
}

impl<
        T1: DimTag,
        T2: DimTag,
        T3: DimTag,
        T4: DimTag,
        T5: DimTag,
        T6: DimTag,
        T7: DimTag,
        T8: DimTag,
    > Shape for Tensor8Shape<T1, T2, T3, T4, T5, T6, T7, T8>
{
    const RANK: u16 = 8;

    fn count(&self) -> usize {
        self.size
    }
}

impl<L: DimTag, R: DimTag> PartialEq<Tensor1Shape<R>> for Tensor1Shape<L> {
    fn eq(&self, other: &Tensor1Shape<R>) -> bool {
        self.d == other.d
    }
}

impl<L1: DimTag, L2: DimTag, R1: DimTag, R2: DimTag> PartialEq<Tensor2Shape<R1, R2>>
    for Tensor2Shape<L1, L2>
{
    fn eq(&self, other: &Tensor2Shape<R1, R2>) -> bool {
        self.d1 == other.d1 && self.d2 == other.d2
    }
}

impl<L1: DimTag, L2: DimTag, L3: DimTag, R1: DimTag, R2: DimTag, R3: DimTag>
    PartialEq<Tensor3Shape<R1, R2, R3>> for Tensor3Shape<L1, L2, L3>
{
    fn eq(&self, other: &Tensor3Shape<R1, R2, R3>) -> bool {
        self.d1 == other.d1 && self.d2 == other.d2 && self.d3 == other.d3
    }
}

impl<
        L1: DimTag,
        L2: DimTag,
        L3: DimTag,
        L4: DimTag,
        R1: DimTag,
        R2: DimTag,
        R3: DimTag,
        R4: DimTag,
    > PartialEq<Tensor4Shape<R1, R2, R3, R4>> for Tensor4Shape<L1, L2, L3, L4>
{
    fn eq(&self, other: &Tensor4Shape<R1, R2, R3, R4>) -> bool {
        self.d1 == other.d1 && self.d2 == other.d2 && self.d3 == other.d3 && self.d4 == other.d4
    }
}

impl<
        L1: DimTag,
        L2: DimTag,
        L3: DimTag,
        L4: DimTag,
        L5: DimTag,
        R1: DimTag,
        R2: DimTag,
        R3: DimTag,
        R4: DimTag,
        R5: DimTag,
    > PartialEq<Tensor5Shape<R1, R2, R3, R4, R5>> for Tensor5Shape<L1, L2, L3, L4, L5>
{
    fn eq(&self, other: &Tensor5Shape<R1, R2, R3, R4, R5>) -> bool {
        self.d1 == other.d1
            && self.d2 == other.d2
            && self.d3 == other.d3
            && self.d4 == other.d4
            && self.d5 == other.d5
    }
}

impl<
        L1: DimTag,
        L2: DimTag,
        L3: DimTag,
        L4: DimTag,
        L5: DimTag,
        L6: DimTag,
        R1: DimTag,
        R2: DimTag,
        R3: DimTag,
        R4: DimTag,
        R5: DimTag,
        R6: DimTag,
    > PartialEq<Tensor6Shape<R1, R2, R3, R4, R5, R6>> for Tensor6Shape<L1, L2, L3, L4, L5, L6>
{
    fn eq(&self, other: &Tensor6Shape<R1, R2, R3, R4, R5, R6>) -> bool {
        self.d1 == other.d1
            && self.d2 == other.d2
            && self.d3 == other.d3
            && self.d4 == other.d4
            && self.d5 == other.d5
            && self.d6 == other.d6
    }
}

impl<
        L1: DimTag,
        L2: DimTag,
        L3: DimTag,
        L4: DimTag,
        L5: DimTag,
        L6: DimTag,
        L7: DimTag,
        R1: DimTag,
        R2: DimTag,
        R3: DimTag,
        R4: DimTag,
        R5: DimTag,
        R6: DimTag,
        R7: DimTag,
    > PartialEq<Tensor7Shape<R1, R2, R3, R4, R5, R6, R7>>
    for Tensor7Shape<L1, L2, L3, L4, L5, L6, L7>
{
    fn eq(&self, other: &Tensor7Shape<R1, R2, R3, R4, R5, R6, R7>) -> bool {
        self.d1 == other.d1
            && self.d2 == other.d2
            && self.d3 == other.d3
            && self.d4 == other.d4
            && self.d5 == other.d5
            && self.d6 == other.d6
            && self.d7 == other.d7
    }
}

impl<
        L1: DimTag,
        L2: DimTag,
        L3: DimTag,
        L4: DimTag,
        L5: DimTag,
        L6: DimTag,
        L7: DimTag,
        L8: DimTag,
        R1: DimTag,
        R2: DimTag,
        R3: DimTag,
        R4: DimTag,
        R5: DimTag,
        R6: DimTag,
        R7: DimTag,
        R8: DimTag,
    > PartialEq<Tensor8Shape<R1, R2, R3, R4, R5, R6, R7, R8>>
    for Tensor8Shape<L1, L2, L3, L4, L5, L6, L7, L8>
{
    fn eq(&self, other: &Tensor8Shape<R1, R2, R3, R4, R5, R6, R7, R8>) -> bool {
        self.d1 == other.d1
            && self.d2 == other.d2
            && self.d3 == other.d3
            && self.d4 == other.d4
            && self.d5 == other.d5
            && self.d6 == other.d6
            && self.d7 == other.d7
            && self.d8 == other.d8
    }
}

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

impl<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag> Debug for Tensor4Shape<T1, T2, T3, T4> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), Error> {
        formatter.write_str("\u{3008}")?;
        self.d1.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d2.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d3.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d4.fmt(formatter)?;
        formatter.write_str("\u{3009}")
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag, T5: DimTag> Debug
    for Tensor5Shape<T1, T2, T3, T4, T5>
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), Error> {
        formatter.write_str("\u{3008}")?;
        self.d1.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d2.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d3.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d4.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d5.fmt(formatter)?;
        formatter.write_str("\u{3009}")
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag, T5: DimTag, T6: DimTag> Debug
    for Tensor6Shape<T1, T2, T3, T4, T5, T6>
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), Error> {
        formatter.write_str("\u{3008}")?;
        self.d1.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d2.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d3.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d4.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d5.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d6.fmt(formatter)?;
        formatter.write_str("\u{3009}")
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag, T5: DimTag, T6: DimTag, T7: DimTag> Debug
    for Tensor7Shape<T1, T2, T3, T4, T5, T6, T7>
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), Error> {
        formatter.write_str("\u{3008}")?;
        self.d1.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d2.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d3.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d4.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d5.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d6.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d7.fmt(formatter)?;
        formatter.write_str("\u{3009}")
    }
}

impl<
        T1: DimTag,
        T2: DimTag,
        T3: DimTag,
        T4: DimTag,
        T5: DimTag,
        T6: DimTag,
        T7: DimTag,
        T8: DimTag,
    > Debug for Tensor8Shape<T1, T2, T3, T4, T5, T6, T7, T8>
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), Error> {
        formatter.write_str("\u{3008}")?;
        self.d1.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d2.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d3.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d4.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d5.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d6.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d7.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d8.fmt(formatter)?;
        formatter.write_str("\u{3009}")
    }
}

pub enum TensorBuilding {}

impl TensorBuilding {
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
    pub fn from_array_to_tensor1<V: Copy + PartialEq + Debug, const N: usize>(
        array: &[V; N],
    ) -> Tensor1<StaticDimTag<N>, V> {
        Tensor1::<StaticDimTag<N>, V> {
            phantom: PhantomData {},
            data: array.to_vec(),
        }
    }
}

#[doc(hidden)]
pub unsafe fn from_vec_to_tensor1_unchecked<T: DimTag, V: Copy + PartialEq + Debug>(
    values: Vec<V>,
) -> Tensor1<T, V> {
    Tensor1::<T, V> {
        phantom: PhantomData {},
        data: values,
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

impl<T1: DimTag, T2: DimTag, T3: DimTag> Tensor3Shape<T1, T2, T3> {
    pub fn with_static<const N: usize>(&self) -> Tensor4Shape<T1, T2, T3, StaticDimTag<N>> {
        Tensor4Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: new_static_dim::<N>(),
            size: self.size.checked_mul(N).expect("Tensor size is too big"),
        }
    }
    pub fn with<T4: DimTag>(&self, dimension: Dim<T4>) -> Tensor4Shape<T1, T2, T3, T4> {
        Tensor4Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: dimension,
            size: self
                .size
                .checked_mul(dimension.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_first(&self) -> Tensor4Shape<T1, T2, T3, T1> {
        Tensor4Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d1,
            size: self
                .size
                .checked_mul(self.d1.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_second(&self) -> Tensor4Shape<T1, T2, T3, T2> {
        Tensor4Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d2,
            size: self
                .size
                .checked_mul(self.d2.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_third(&self) -> Tensor4Shape<T1, T2, T3, T3> {
        Tensor4Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d3,
            size: self
                .size
                .checked_mul(self.d2.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn switch_12(&self) -> Tensor3Shape<T2, T1, T3> {
        Tensor3Shape {
            d1: self.d2,
            d2: self.d1,
            d3: self.d3,
            size: self.size,
        }
    }
    pub fn switch_13(&self) -> Tensor3Shape<T3, T2, T1> {
        Tensor3Shape {
            d1: self.d3,
            d2: self.d2,
            d3: self.d1,
            size: self.size,
        }
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag> Tensor4Shape<T1, T2, T3, T4> {
    pub fn with_static<const N: usize>(&self) -> Tensor5Shape<T1, T2, T3, T4, StaticDimTag<N>> {
        Tensor5Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: new_static_dim::<N>(),
            size: self.size.checked_mul(N).expect("Tensor size is too big"),
        }
    }
    pub fn with<T5: DimTag>(&self, dimension: Dim<T5>) -> Tensor5Shape<T1, T2, T3, T4, T5> {
        Tensor5Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: dimension,
            size: self
                .size
                .checked_mul(dimension.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_first(&self) -> Tensor5Shape<T1, T2, T3, T4, T1> {
        Tensor5Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d1,
            size: self
                .size
                .checked_mul(self.d1.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_second(&self) -> Tensor5Shape<T1, T2, T3, T4, T2> {
        Tensor5Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d2,
            size: self
                .size
                .checked_mul(self.d1.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_third(&self) -> Tensor5Shape<T1, T2, T3, T4, T3> {
        Tensor5Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d3,
            size: self
                .size
                .checked_mul(self.d1.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_fourth(&self) -> Tensor5Shape<T1, T2, T3, T4, T4> {
        Tensor5Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d4,
            size: self
                .size
                .checked_mul(self.d1.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn switch_12(&self) -> Tensor4Shape<T2, T1, T3, T4> {
        Tensor4Shape {
            d1: self.d2,
            d2: self.d1,
            d3: self.d3,
            d4: self.d4,
            size: self.size,
        }
    }
    pub fn switch_13(&self) -> Tensor4Shape<T3, T2, T1, T4> {
        Tensor4Shape {
            d1: self.d3,
            d2: self.d2,
            d3: self.d1,
            d4: self.d4,
            size: self.size,
        }
    }
    pub fn switch_14(&self) -> Tensor4Shape<T4, T2, T3, T1> {
        Tensor4Shape {
            d1: self.d4,
            d2: self.d2,
            d3: self.d3,
            d4: self.d1,
            size: self.size,
        }
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag, T5: DimTag> Tensor5Shape<T1, T2, T3, T4, T5> {
    pub fn with_static<const N: usize>(&self) -> Tensor6Shape<T1, T2, T3, T4, T5, StaticDimTag<N>> {
        Tensor6Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: new_static_dim::<N>(),
            size: self.size.checked_mul(N).expect("Tensor size is too big"),
        }
    }
    pub fn with<T6: DimTag>(&self, dimension: Dim<T6>) -> Tensor6Shape<T1, T2, T3, T4, T5, T6> {
        Tensor6Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: dimension,
            size: self
                .size
                .checked_mul(dimension.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_first(&self) -> Tensor6Shape<T1, T2, T3, T4, T5, T1> {
        Tensor6Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d1,
            size: self
                .size
                .checked_mul(self.d1.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_second(&self) -> Tensor6Shape<T1, T2, T3, T4, T5, T2> {
        Tensor6Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d2,
            size: self
                .size
                .checked_mul(self.d1.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_third(&self) -> Tensor6Shape<T1, T2, T3, T4, T5, T3> {
        Tensor6Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d3,
            size: self
                .size
                .checked_mul(self.d1.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_fourth(&self) -> Tensor6Shape<T1, T2, T3, T4, T5, T4> {
        Tensor6Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d4,
            size: self
                .size
                .checked_mul(self.d1.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_fifth(&self) -> Tensor6Shape<T1, T2, T3, T4, T5, T5> {
        Tensor6Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d5,
            size: self
                .size
                .checked_mul(self.d1.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn switch_12(&self) -> Tensor5Shape<T2, T1, T3, T4, T5> {
        Tensor5Shape {
            d1: self.d2,
            d2: self.d1,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            size: self.size,
        }
    }
    pub fn switch_13(&self) -> Tensor5Shape<T3, T2, T1, T4, T5> {
        Tensor5Shape {
            d1: self.d3,
            d2: self.d2,
            d3: self.d1,
            d4: self.d4,
            d5: self.d5,
            size: self.size,
        }
    }
    pub fn switch_14(&self) -> Tensor5Shape<T4, T2, T3, T1, T5> {
        Tensor5Shape {
            d1: self.d4,
            d2: self.d2,
            d3: self.d3,
            d4: self.d1,
            d5: self.d5,
            size: self.size,
        }
    }
    pub fn switch_15(&self) -> Tensor5Shape<T5, T2, T3, T4, T1> {
        Tensor5Shape {
            d1: self.d5,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d1,
            size: self.size,
        }
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag, T5: DimTag, T6: DimTag>
    Tensor6Shape<T1, T2, T3, T4, T5, T6>
{
    pub fn with_static<const N: usize>(
        &self,
    ) -> Tensor7Shape<T1, T2, T3, T4, T5, T6, StaticDimTag<N>> {
        Tensor7Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d6,
            d7: new_static_dim::<N>(),
            size: self.size.checked_mul(N).expect("Tensor size is too big"),
        }
    }
    pub fn with<T7: DimTag>(&self, dimension: Dim<T7>) -> Tensor7Shape<T1, T2, T3, T4, T5, T6, T7> {
        Tensor7Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d6,
            d7: dimension,
            size: self
                .size
                .checked_mul(dimension.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_first(&self) -> Tensor7Shape<T1, T2, T3, T4, T5, T6, T1> {
        Tensor7Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d6,
            d7: self.d1,
            size: self
                .size
                .checked_mul(self.d1.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_second(&self) -> Tensor7Shape<T1, T2, T3, T4, T5, T6, T2> {
        Tensor7Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d6,
            d7: self.d2,
            size: self
                .size
                .checked_mul(self.d1.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_third(&self) -> Tensor7Shape<T1, T2, T3, T4, T5, T6, T3> {
        Tensor7Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d6,
            d7: self.d3,
            size: self
                .size
                .checked_mul(self.d1.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_fourth(&self) -> Tensor7Shape<T1, T2, T3, T4, T5, T6, T4> {
        Tensor7Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d6,
            d7: self.d4,
            size: self
                .size
                .checked_mul(self.d1.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_fifth(&self) -> Tensor7Shape<T1, T2, T3, T4, T5, T6, T5> {
        Tensor7Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d6,
            d7: self.d5,
            size: self
                .size
                .checked_mul(self.d1.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_sixth(&self) -> Tensor7Shape<T1, T2, T3, T4, T5, T6, T6> {
        Tensor7Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d6,
            d7: self.d6,
            size: self
                .size
                .checked_mul(self.d1.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn switch_12(&self) -> Tensor6Shape<T2, T1, T3, T4, T5, T6> {
        Tensor6Shape {
            d1: self.d2,
            d2: self.d1,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d6,
            size: self.size,
        }
    }
    pub fn switch_13(&self) -> Tensor6Shape<T3, T2, T1, T4, T5, T6> {
        Tensor6Shape {
            d1: self.d3,
            d2: self.d2,
            d3: self.d1,
            d4: self.d4,
            d5: self.d5,
            d6: self.d6,
            size: self.size,
        }
    }
    pub fn switch_14(&self) -> Tensor6Shape<T4, T2, T3, T1, T5, T6> {
        Tensor6Shape {
            d1: self.d4,
            d2: self.d2,
            d3: self.d3,
            d4: self.d1,
            d5: self.d5,
            d6: self.d6,
            size: self.size,
        }
    }
    pub fn switch_15(&self) -> Tensor6Shape<T5, T2, T3, T4, T1, T6> {
        Tensor6Shape {
            d1: self.d5,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d1,
            d6: self.d6,
            size: self.size,
        }
    }
    pub fn switch_16(&self) -> Tensor6Shape<T6, T2, T3, T4, T5, T1> {
        Tensor6Shape {
            d1: self.d6,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d1,
            size: self.size,
        }
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag, T5: DimTag, T6: DimTag, T7: DimTag>
    Tensor7Shape<T1, T2, T3, T4, T5, T6, T7>
{
    pub fn with_static<const N: usize>(
        &self,
    ) -> Tensor8Shape<T1, T2, T3, T4, T5, T6, T7, StaticDimTag<N>> {
        Tensor8Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d6,
            d7: self.d7,
            d8: new_static_dim::<N>(),
            size: self.size.checked_mul(N).expect("Tensor size is too big"),
        }
    }
    pub fn with<T8: DimTag>(
        &self,
        dimension: Dim<T8>,
    ) -> Tensor8Shape<T1, T2, T3, T4, T5, T6, T7, T8> {
        Tensor8Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d6,
            d7: self.d7,
            d8: dimension,
            size: self
                .size
                .checked_mul(dimension.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_first(&self) -> Tensor8Shape<T1, T2, T3, T4, T5, T6, T7, T1> {
        Tensor8Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d6,
            d7: self.d7,
            d8: self.d1,
            size: self
                .size
                .checked_mul(self.d1.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_second(&self) -> Tensor8Shape<T1, T2, T3, T4, T5, T6, T7, T2> {
        Tensor8Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d6,
            d7: self.d7,
            d8: self.d2,
            size: self
                .size
                .checked_mul(self.d1.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_third(&self) -> Tensor8Shape<T1, T2, T3, T4, T5, T6, T7, T3> {
        Tensor8Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d6,
            d7: self.d7,
            d8: self.d3,
            size: self
                .size
                .checked_mul(self.d1.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_fourth(&self) -> Tensor8Shape<T1, T2, T3, T4, T5, T6, T7, T4> {
        Tensor8Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d6,
            d7: self.d7,
            d8: self.d4,
            size: self
                .size
                .checked_mul(self.d1.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_fifth(&self) -> Tensor8Shape<T1, T2, T3, T4, T5, T6, T7, T5> {
        Tensor8Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d6,
            d7: self.d7,
            d8: self.d5,
            size: self
                .size
                .checked_mul(self.d1.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_sixth(&self) -> Tensor8Shape<T1, T2, T3, T4, T5, T6, T7, T6> {
        Tensor8Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d6,
            d7: self.d7,
            d8: self.d6,
            size: self
                .size
                .checked_mul(self.d1.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn with_seventh(&self) -> Tensor8Shape<T1, T2, T3, T4, T5, T6, T7, T7> {
        Tensor8Shape {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d6,
            d7: self.d7,
            d8: self.d7,
            size: self
                .size
                .checked_mul(self.d1.as_usize())
                .expect("Tensor size is too big"),
        }
    }
    pub fn switch_12(&self) -> Tensor7Shape<T2, T1, T3, T4, T5, T6, T7> {
        Tensor7Shape {
            d1: self.d2,
            d2: self.d1,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d6,
            d7: self.d7,
            size: self.size,
        }
    }
    pub fn switch_13(&self) -> Tensor7Shape<T3, T2, T1, T4, T5, T6, T7> {
        Tensor7Shape {
            d1: self.d3,
            d2: self.d2,
            d3: self.d1,
            d4: self.d4,
            d5: self.d5,
            d6: self.d6,
            d7: self.d7,
            size: self.size,
        }
    }
    pub fn switch_14(&self) -> Tensor7Shape<T4, T2, T3, T1, T5, T6, T7> {
        Tensor7Shape {
            d1: self.d4,
            d2: self.d2,
            d3: self.d3,
            d4: self.d1,
            d5: self.d5,
            d6: self.d6,
            d7: self.d7,
            size: self.size,
        }
    }
    pub fn switch_15(&self) -> Tensor7Shape<T5, T2, T3, T4, T1, T6, T7> {
        Tensor7Shape {
            d1: self.d5,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d1,
            d6: self.d6,
            d7: self.d7,
            size: self.size,
        }
    }
    pub fn switch_16(&self) -> Tensor7Shape<T6, T2, T3, T4, T5, T1, T7> {
        Tensor7Shape {
            d1: self.d6,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d1,
            d7: self.d7,
            size: self.size,
        }
    }
    pub fn switch_17(&self) -> Tensor7Shape<T7, T2, T3, T4, T5, T6, T1> {
        Tensor7Shape {
            d1: self.d7,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d6,
            d7: self.d1,
            size: self.size,
        }
    }
}

pub struct TensorPreparation<T: Tensor> {
    expected_size: usize,
    shape: T::Shape,
    data: Vec<T::Element>,
    generator: fn(T::Shape, Vec<T::Element>) -> T,
}

impl<T: Tensor> TensorPreparation<T> {
    pub fn count_set_elements(&self) -> usize {
        self.data.len()
    }
    pub fn count_unset_elements(&self) -> usize {
        self.expected_size - self.data.len()
    }
    pub fn append_vec(mut self, values: &mut Vec<T::Element>) -> Self {
        self.data.append(values);
        self.data.truncate(self.expected_size);
        self
    }
    pub fn append_array<const N: usize>(mut self, values: [T::Element; N]) -> Self {
        self.data.append(&mut Vec::<_>::from(values));
        self.data.truncate(self.expected_size);
        self
    }
    pub fn fill(mut self, value: &T::Element) -> T
    where
        T::Element: Clone,
    {
        let missing_elements_count = self.expected_size - self.data.len();
        self.data
            .append(&mut vec![value.clone(); missing_elements_count]);
        self.generate()
    }
    pub fn fill_during(mut self, value: &T::Element, additional_count: usize) -> Self
    where
        T::Element: Clone,
    {
        let missing_elements_count = (self.expected_size - self.data.len()).min(additional_count);
        self.data
            .append(&mut vec![value.clone(); missing_elements_count]);
        self
    }
    pub fn generate(self) -> T {
        if self.data.len() != self.expected_size {
            panic!("Invalid size")
        }
        (self.generator)(self.shape, self.data)
    }
    pub fn try_generate(self) -> Result<T, Self> {
        if self.data.len() != self.expected_size {
            Err(self)
        } else {
            Ok((self.generator)(self.shape, self.data))
        }
    }
}

pub trait TensorBuilder<V> {
    type Tensor: Tensor;
    type MultiIndex: Eq + Debug + Copy;
    fn fill(&self, value: &V) -> Self::Tensor
    where
        V: Clone;
    fn define(&self, f: impl FnMut(Self::MultiIndex) -> V) -> Self::Tensor;
    fn prepare(&self) -> TensorPreparation<Self::Tensor>;
}

impl<T: DimTag, V: PartialEq + Debug> TensorBuilder<V> for Tensor1Shape<T> {
    type Tensor = Tensor1<T, V>;
    type MultiIndex = (usize,);

    fn fill(&self, value: &V) -> Self::Tensor
    where
        V: Clone,
    {
        Self::Tensor {
            phantom: PhantomData,
            data: vec![value.clone(); self.count()],
        }
    }
    fn define(&self, mut f: impl FnMut(Self::MultiIndex) -> V) -> Self::Tensor {
        let mut data = Vec::<V>::with_capacity(self.d.as_usize());
        for i in 0..self.d.as_usize() {
            data.push(f((i,)))
        }
        Self::Tensor {
            phantom: PhantomData,
            data,
        }
    }
    fn prepare(&self) -> TensorPreparation<Self::Tensor> {
        TensorPreparation {
            expected_size: self.count(),
            data: Vec::<V>::with_capacity(self.count()),
            shape: (self.d,),
            generator: |_dims, data| Self::Tensor {
                phantom: PhantomData,
                data,
            },
        }
    }
}

impl<T1: DimTag, T2: DimTag, V: PartialEq + Debug> TensorBuilder<V> for Tensor2Shape<T1, T2> {
    type Tensor = Tensor2<T1, T2, V>;
    type MultiIndex = (usize, usize);

    fn fill(&self, value: &V) -> Self::Tensor
    where
        V: Clone,
    {
        Self::Tensor {
            d1: self.d1,
            d2: self.d2,
            data: vec![value.clone(); self.count()],
        }
    }
    fn define(&self, mut f: impl FnMut(Self::MultiIndex) -> V) -> Self::Tensor {
        let mut data = Vec::<V>::with_capacity(self.count());
        for i1 in 0..self.d1.as_usize() {
            for i2 in 0..self.d2.as_usize() {
                data.push(f((i1, i2)));
            }
        }
        Self::Tensor {
            d1: self.d1,
            d2: self.d2,
            data,
        }
    }
    fn prepare(&self) -> TensorPreparation<Self::Tensor> {
        TensorPreparation {
            expected_size: self.count(),
            data: Vec::<V>::with_capacity(self.count()),
            shape: (self.d1, self.d2),
            generator: |shape, data| Self::Tensor {
                d1: shape.0,
                d2: shape.1,
                data,
            },
        }
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, V: PartialEq + Debug> TensorBuilder<V>
    for Tensor3Shape<T1, T2, T3>
{
    type Tensor = Tensor3<T1, T2, T3, V>;
    type MultiIndex = (usize, usize, usize);

    fn fill(&self, value: &V) -> Self::Tensor
    where
        V: Clone,
    {
        Self::Tensor {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            data: vec![value.clone(); self.count()],
        }
    }
    fn define(&self, mut f: impl FnMut(Self::MultiIndex) -> V) -> Self::Tensor {
        let mut data = Vec::<V>::with_capacity(self.count());
        for i1 in 0..self.d1.as_usize() {
            for i2 in 0..self.d2.as_usize() {
                for i3 in 0..self.d3.as_usize() {
                    data.push(f((i1, i2, i3)));
                }
            }
        }
        Self::Tensor {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            data,
        }
    }
    fn prepare(&self) -> TensorPreparation<Self::Tensor> {
        TensorPreparation {
            expected_size: self.count(),
            data: Vec::<V>::with_capacity(self.count()),
            shape: (self.d1, self.d2, self.d3),
            generator: |shape, data| Self::Tensor {
                d1: shape.0,
                d2: shape.1,
                d3: shape.2,
                data,
            },
        }
    }
}
