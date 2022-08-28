//! The mathematical word [tensor](https://en.wikipedia.org/wiki/Tensor) is used here to describe more broadly
//! arrays with multiple indexes.
//!
//!

use super::dimensions::*;
use std::fmt::{Debug, Error, Formatter};
use std::marker::{Copy, PhantomData};
use std::ops::{Index, IndexMut};

/// A general trait shared by all tensor types.
pub trait Tensor: PartialEq + Index<Self::MultiIndex> + IndexMut<Self::MultiIndex> + Debug {
    /// The type of elements stored in the tensor.
    type Element: PartialEq + Debug;
    /// The type of the dimensions tuple
    type Shape: Copy + Eq + Debug;
    // the type of a multi-index to retrieve an element
    type MultiIndex: Copy + Eq + Debug;
    /// The number of dimensions, also corresponding
    /// to the number of indexes.
    const RANK: u16;
    /// The total number of coordinates
    fn count(&self) -> u64;
    /// The tuple of dimensions
    fn shape(&self) -> Self::Shape;
    /// Update some elements of the tensor
    fn update(&mut self, updater: impl FnMut(Self::MultiIndex, &mut Self::Element));
}

/// A tensor of rank 0 (that is to say a scalar) containing a single value of type V.
#[derive(PartialEq)]
pub struct Tensor0<V: PartialEq + Debug> {
    pub(crate) data: V,
}

/// A tensor of rank 1 and dimension represented by tag T containing values of type V.
#[derive(PartialEq)]
pub struct Tensor1<T: DimTag, V: PartialEq + Debug> {
    pub(crate) phantom: PhantomData<T>,
    pub(crate) data: Vec<V>,
}

#[derive(PartialEq)]
pub struct Tensor2<T1: DimTag, T2: DimTag, V: PartialEq + Debug> {
    pub(crate) d1: Dim<T1>,
    pub(crate) d2: Dim<T2>,
    pub(crate) data: Vec<V>,
}

#[derive(PartialEq)]
pub struct Tensor3<T1: DimTag, T2: DimTag, T3: DimTag, V: PartialEq + Debug> {
    pub(crate) d1: Dim<T1>,
    pub(crate) d2: Dim<T2>,
    pub(crate) d3: Dim<T3>,
    pub(crate) data: Vec<V>,
}

#[derive(PartialEq)]
pub struct Tensor4<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag, V: PartialEq + Debug> {
    pub(crate) d1: Dim<T1>,
    pub(crate) d2: Dim<T2>,
    pub(crate) d3: Dim<T3>,
    pub(crate) d4: Dim<T4>,
    pub(crate) data: Vec<V>,
}

#[derive(PartialEq)]
pub struct Tensor5<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag, T5: DimTag, V: PartialEq + Debug>
{
    pub(crate) d1: Dim<T1>,
    pub(crate) d2: Dim<T2>,
    pub(crate) d3: Dim<T3>,
    pub(crate) d4: Dim<T4>,
    pub(crate) d5: Dim<T5>,
    pub(crate) data: Vec<V>,
}

#[derive(PartialEq)]
pub struct Tensor6<
    T1: DimTag,
    T2: DimTag,
    T3: DimTag,
    T4: DimTag,
    T5: DimTag,
    T6: DimTag,
    V: PartialEq + Debug,
> {
    pub(crate) d1: Dim<T1>,
    pub(crate) d2: Dim<T2>,
    pub(crate) d3: Dim<T3>,
    pub(crate) d4: Dim<T4>,
    pub(crate) d5: Dim<T5>,
    pub(crate) d6: Dim<T6>,
    pub(crate) data: Vec<V>,
}

#[derive(PartialEq)]
pub struct Tensor7<
    T1: DimTag,
    T2: DimTag,
    T3: DimTag,
    T4: DimTag,
    T5: DimTag,
    T6: DimTag,
    T7: DimTag,
    V: PartialEq + Debug,
> {
    pub(crate) d1: Dim<T1>,
    pub(crate) d2: Dim<T2>,
    pub(crate) d3: Dim<T3>,
    pub(crate) d4: Dim<T4>,
    pub(crate) d5: Dim<T5>,
    pub(crate) d6: Dim<T6>,
    pub(crate) d7: Dim<T7>,
    pub(crate) data: Vec<V>,
}

#[derive(PartialEq)]
pub struct Tensor8<
    T1: DimTag,
    T2: DimTag,
    T3: DimTag,
    T4: DimTag,
    T5: DimTag,
    T6: DimTag,
    T7: DimTag,
    T8: DimTag,
    V: PartialEq + Debug,
> {
    pub(crate) d1: Dim<T1>,
    pub(crate) d2: Dim<T2>,
    pub(crate) d3: Dim<T3>,
    pub(crate) d4: Dim<T4>,
    pub(crate) d5: Dim<T5>,
    pub(crate) d6: Dim<T6>,
    pub(crate) d7: Dim<T7>,
    pub(crate) d8: Dim<T8>,
    pub(crate) data: Vec<V>,
}

pub type Scalar<V> = Tensor0<V>;
pub type Vector<T, V> = Tensor1<T, V>;
pub type Matrix<T1, T2, V> = Tensor2<T1, T2, V>;

pub type StaticVector<const N: usize, V> = Tensor1<StaticDimTag<N>, V>;
pub type StaticMatrix<const N1: usize, const N2: usize, V> =
    Tensor2<StaticDimTag<N1>, StaticDimTag<N2>, V>;

impl<V: PartialEq + Debug> Tensor for Tensor0<V> {
    type Element = V;
    type Shape = ();
    type MultiIndex = ();
    const RANK: u16 = 0;
    fn count(&self) -> u64 {
        1
    }
    fn shape(&self) -> Self::Shape {}
    fn update(&mut self, mut updater: impl FnMut((), &mut V)) {
        updater((), &mut self.data)
    }
}

impl<T: DimTag, V: PartialEq + Debug> Tensor for Tensor1<T, V> {
    type Element = V;
    type Shape = (Dim<T>,);
    type MultiIndex = (usize,);
    const RANK: u16 = 1;
    fn count(&self) -> u64 {
        self.data.len() as u64
    }
    fn shape(&self) -> Self::Shape {
        let n = self.data.len();
        unsafe { (Dim::<T>::unsafe_new(n),) }
    }
    fn update(&mut self, mut updater: impl FnMut(Self::MultiIndex, &mut V)) {
        for (i, element) in self.data.iter_mut().enumerate() {
            updater((i,), element)
        }
    }
}

impl<T1: DimTag, T2: DimTag, V: PartialEq + Debug> Tensor for Tensor2<T1, T2, V> {
    type Element = V;
    type Shape = (Dim<T1>, Dim<T2>);
    type MultiIndex = (usize, usize);
    const RANK: u16 = 2;
    fn count(&self) -> u64 {
        self.data.len() as u64
    }
    fn shape(&self) -> Self::Shape {
        (self.d1, self.d2)
    }
    fn update(&mut self, mut updater: impl FnMut(Self::MultiIndex, &mut V)) {
        let d2 = self.d2.as_usize();
        for (i, element) in self.data.iter_mut().enumerate() {
            let i2 = i % d2;
            let i1 = i / d2;
            updater((i1, i2), element)
        }
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, V: PartialEq + Debug> Tensor for Tensor3<T1, T2, T3, V> {
    type Element = V;
    type Shape = (Dim<T1>, Dim<T2>, Dim<T3>);
    type MultiIndex = (usize, usize, usize);
    const RANK: u16 = 3;
    fn count(&self) -> u64 {
        self.data.len() as u64
    }
    fn shape(&self) -> Self::Shape {
        (self.d1, self.d2, self.d3)
    }
    fn update(&mut self, mut updater: impl FnMut((usize, usize, usize), &mut V)) {
        let d2 = self.d2.as_usize();
        let d3 = self.d3.as_usize();
        for (j, element) in self.data.iter_mut().enumerate() {
            let i3 = j % d3;
            let j = j / d3;
            let i2 = j % d2;
            let i1 = j / d2;
            updater((i1, i2, i3), element)
        }
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag, V: PartialEq + Debug> Tensor
    for Tensor4<T1, T2, T3, T4, V>
{
    type Element = V;
    type Shape = (Dim<T1>, Dim<T2>, Dim<T3>, Dim<T4>);
    type MultiIndex = (usize, usize, usize, usize);
    const RANK: u16 = 4;
    fn count(&self) -> u64 {
        self.data.len() as u64
    }
    fn shape(&self) -> Self::Shape {
        (self.d1, self.d2, self.d3, self.d4)
    }
    fn update(&mut self, mut updater: impl FnMut(Self::MultiIndex, &mut V)) {
        let d2 = self.d2.as_usize();
        let d3 = self.d3.as_usize();
        let d4 = self.d4.as_usize();
        for (j, element) in self.data.iter_mut().enumerate() {
            let i4 = j % d4;
            let j = j / d4;
            let i3 = j % d3;
            let j = j / d3;
            let i2 = j % d2;
            let i1 = j / d2;
            updater((i1, i2, i3, i4), element)
        }
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag, T5: DimTag, V: PartialEq + Debug> Tensor
    for Tensor5<T1, T2, T3, T4, T5, V>
{
    type Element = V;
    type Shape = (Dim<T1>, Dim<T2>, Dim<T3>, Dim<T4>, Dim<T5>);
    type MultiIndex = (usize, usize, usize, usize, usize);
    const RANK: u16 = 5;
    fn count(&self) -> u64 {
        self.data.len() as u64
    }
    fn shape(&self) -> Self::Shape {
        (self.d1, self.d2, self.d3, self.d4, self.d5)
    }
    fn update(&mut self, mut updater: impl FnMut(Self::MultiIndex, &mut V)) {
        let d2 = self.d2.as_usize();
        let d3 = self.d3.as_usize();
        let d4 = self.d4.as_usize();
        let d5 = self.d5.as_usize();
        for (j, element) in self.data.iter_mut().enumerate() {
            let i5 = j % d5;
            let j = j / d5;
            let i4 = j % d4;
            let j = j / d4;
            let i3 = j % d3;
            let j = j / d3;
            let i2 = j % d2;
            let i1 = j / d2;
            updater((i1, i2, i3, i4, i5), element)
        }
    }
}

impl<
        T1: DimTag,
        T2: DimTag,
        T3: DimTag,
        T4: DimTag,
        T5: DimTag,
        T6: DimTag,
        V: PartialEq + Debug,
    > Tensor for Tensor6<T1, T2, T3, T4, T5, T6, V>
{
    type Element = V;
    type Shape = (Dim<T1>, Dim<T2>, Dim<T3>, Dim<T4>, Dim<T5>, Dim<T6>);
    type MultiIndex = (usize, usize, usize, usize, usize, usize);
    const RANK: u16 = 6;
    fn count(&self) -> u64 {
        self.data.len() as u64
    }
    fn shape(&self) -> Self::Shape {
        (self.d1, self.d2, self.d3, self.d4, self.d5, self.d6)
    }
    fn update(&mut self, mut updater: impl FnMut(Self::MultiIndex, &mut V)) {
        let d2 = self.d2.as_usize();
        let d3 = self.d3.as_usize();
        let d4 = self.d4.as_usize();
        let d5 = self.d5.as_usize();
        let d6 = self.d6.as_usize();
        for (j, element) in self.data.iter_mut().enumerate() {
            let i6 = j % d6;
            let j = j / d6;
            let i5 = j % d5;
            let j = j / d5;
            let i4 = j % d4;
            let j = j / d4;
            let i3 = j % d3;
            let j = j / d3;
            let i2 = j % d2;
            let i1 = j / d2;
            updater((i1, i2, i3, i4, i5, i6), element)
        }
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
        V: PartialEq + Debug,
    > Tensor for Tensor7<T1, T2, T3, T4, T5, T6, T7, V>
{
    type Element = V;
    type Shape = (
        Dim<T1>,
        Dim<T2>,
        Dim<T3>,
        Dim<T4>,
        Dim<T5>,
        Dim<T6>,
        Dim<T7>,
    );
    type MultiIndex = (usize, usize, usize, usize, usize, usize, usize);
    const RANK: u16 = 7;
    fn count(&self) -> u64 {
        self.data.len() as u64
    }
    fn shape(&self) -> Self::Shape {
        (
            self.d1, self.d2, self.d3, self.d4, self.d5, self.d6, self.d7,
        )
    }
    fn update(&mut self, mut updater: impl FnMut(Self::MultiIndex, &mut V)) {
        let d2 = self.d2.as_usize();
        let d3 = self.d3.as_usize();
        let d4 = self.d4.as_usize();
        let d5 = self.d5.as_usize();
        let d6 = self.d6.as_usize();
        let d7 = self.d7.as_usize();
        for (j, element) in self.data.iter_mut().enumerate() {
            let i7 = j % d7;
            let j = j / d7;
            let i6 = j % d6;
            let j = j / d6;
            let i5 = j % d5;
            let j = j / d5;
            let i4 = j % d4;
            let j = j / d4;
            let i3 = j % d3;
            let j = j / d3;
            let i2 = j % d2;
            let i1 = j / d2;
            updater((i1, i2, i3, i4, i5, i6, i7), element)
        }
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
        V: PartialEq + Debug,
    > Tensor for Tensor8<T1, T2, T3, T4, T5, T6, T7, T8, V>
{
    type Element = V;
    type Shape = (
        Dim<T1>,
        Dim<T2>,
        Dim<T3>,
        Dim<T4>,
        Dim<T5>,
        Dim<T6>,
        Dim<T7>,
        Dim<T8>,
    );
    type MultiIndex = (usize, usize, usize, usize, usize, usize, usize, usize);
    const RANK: u16 = 8;
    fn count(&self) -> u64 {
        self.data.len() as u64
    }
    fn shape(&self) -> Self::Shape {
        (
            self.d1, self.d2, self.d3, self.d4, self.d5, self.d6, self.d7, self.d8,
        )
    }
    fn update(&mut self, mut updater: impl FnMut(Self::MultiIndex, &mut V)) {
        let d2 = self.d2.as_usize();
        let d3 = self.d3.as_usize();
        let d4 = self.d4.as_usize();
        let d5 = self.d5.as_usize();
        let d6 = self.d6.as_usize();
        let d7 = self.d7.as_usize();
        let d8 = self.d8.as_usize();
        for (j, element) in self.data.iter_mut().enumerate() {
            let i8 = j % d8;
            let j = j / d8;
            let i7 = j % d7;
            let j = j / d7;
            let i6 = j % d6;
            let j = j / d6;
            let i5 = j % d5;
            let j = j / d5;
            let i4 = j % d4;
            let j = j / d4;
            let i3 = j % d3;
            let j = j / d3;
            let i2 = j % d2;
            let i1 = j / d2;
            updater((i1, i2, i3, i4, i5, i6, i7, i8), element)
        }
    }
}

impl<V: Clone + PartialEq + Debug> Clone for Tensor0<V> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
        }
    }
}

impl<V: Copy + PartialEq + Debug> Copy for Tensor0<V> {}

impl<T: DimTag, V: Clone + PartialEq + Debug> Clone for Tensor1<T, V> {
    fn clone(&self) -> Self {
        Self {
            phantom: PhantomData {},
            data: self.data.clone(),
        }
    }
}

impl<T1: DimTag, T2: DimTag, V: Clone + PartialEq + Debug> Clone for Tensor2<T1, T2, V> {
    fn clone(&self) -> Self {
        Self {
            d1: self.d1,
            d2: self.d2,
            data: self.data.clone(),
        }
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, V: Clone + PartialEq + Debug> Clone
    for Tensor3<T1, T2, T3, V>
{
    fn clone(&self) -> Self {
        Self {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            data: self.data.clone(),
        }
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag, V: Clone + PartialEq + Debug> Clone
    for Tensor4<T1, T2, T3, T4, V>
{
    fn clone(&self) -> Self {
        Self {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            data: self.data.clone(),
        }
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag, T5: DimTag, V: Clone + PartialEq + Debug> Clone
    for Tensor5<T1, T2, T3, T4, T5, V>
{
    fn clone(&self) -> Self {
        Self {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            data: self.data.clone(),
        }
    }
}

impl<
        T1: DimTag,
        T2: DimTag,
        T3: DimTag,
        T4: DimTag,
        T5: DimTag,
        T6: DimTag,
        V: Clone + PartialEq + Debug,
    > Clone for Tensor6<T1, T2, T3, T4, T5, T6, V>
{
    fn clone(&self) -> Self {
        Self {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d6,
            data: self.data.clone(),
        }
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
        V: Clone + PartialEq + Debug,
    > Clone for Tensor7<T1, T2, T3, T4, T5, T6, T7, V>
{
    fn clone(&self) -> Self {
        Self {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d6,
            d7: self.d7,
            data: self.data.clone(),
        }
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
        V: Clone + PartialEq + Debug,
    > Clone for Tensor8<T1, T2, T3, T4, T5, T6, T7, T8, V>
{
    fn clone(&self) -> Self {
        Self {
            d1: self.d1,
            d2: self.d2,
            d3: self.d3,
            d4: self.d4,
            d5: self.d5,
            d6: self.d6,
            d7: self.d7,
            d8: self.d8,
            data: self.data.clone(),
        }
    }
}

const FORMATED_ELEMENT_MAX_COUNT: usize = 10;

impl<V: PartialEq + Debug> Debug for Tensor0<V> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), Error> {
        formatter.write_str("\u{3008}\u{3009}[")?;
        self.data.fmt(formatter)?;
        formatter.write_str("]")
    }
}

#[inline]
fn display<'a, V: 'a + Debug>(
    iter: impl Iterator<Item = &'a V>,
    (d4, d3, d2): (usize, usize, usize),
    formatter: &mut Formatter<'_>,
) -> Result<(), Error> {
    for (i, elt) in iter.enumerate() {
        if i > 0 {
            if d4 != 0 && i % d4 == 0 {
                formatter.write_str(" ||| ")?
            } else if d3 != 0 && i % d3 == 0 {
                formatter.write_str(" || ")?
            } else if d2 != 0 && i % d2 == 0 {
                formatter.write_str(" | ")?
            } else {
                formatter.write_str(", ")?
            }
        }
        elt.fmt(formatter)?;
    }
    Ok(())
}

impl<T: DimTag, V: PartialEq + Debug> Debug for Tensor1<T, V> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), Error> {
        let d = unsafe { Dim::<T>::unsafe_new(self.data.len()) };
        formatter.write_str("\u{3008}")?;
        d.fmt(formatter)?;
        formatter.write_str("\u{3009}[")?;

        let show_all = formatter.alternate();

        if show_all {
            display(self.data.iter(), (0, 0, 0), formatter)?
        } else {
            display(
                self.data.iter().take(FORMATED_ELEMENT_MAX_COUNT),
                (0, 0, 0),
                formatter,
            )?
        }

        if show_all || self.data.len() <= FORMATED_ELEMENT_MAX_COUNT {
            formatter.write_str("]")
        } else {
            formatter.write_str(", \u{2026}]")
        }
    }
}

impl<T1: DimTag, T2: DimTag, V: PartialEq + Debug> Debug for Tensor2<T1, T2, V> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), Error> {
        formatter.write_str("\u{3008}")?;
        self.d1.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d2.fmt(formatter)?;
        formatter.write_str("\u{3009}[")?;

        let show_all = formatter.alternate();
        let d2 = self.d2.as_usize();

        if show_all {
            display(self.data.iter(), (0, 0, d2), formatter)?
        } else {
            display(
                self.data.iter().take(FORMATED_ELEMENT_MAX_COUNT),
                (0, 0, d2),
                formatter,
            )?
        }

        if show_all || self.data.len() <= FORMATED_ELEMENT_MAX_COUNT {
            formatter.write_str("]")
        } else {
            formatter.write_str(", \u{2026}]")
        }
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, V: PartialEq + Debug> Debug for Tensor3<T1, T2, T3, V> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), Error> {
        formatter.write_str("\u{3008}")?;
        self.d1.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d2.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.d3.fmt(formatter)?;
        formatter.write_str("\u{3009}[")?;

        let show_all = formatter.alternate();
        let d2 = self.d2.as_usize();
        let d3 = self.d3.as_usize() * d2;

        if show_all {
            display(self.data.iter(), (0, d3, d2), formatter)?
        } else {
            display(
                self.data.iter().take(FORMATED_ELEMENT_MAX_COUNT),
                (0, d3, d2),
                formatter,
            )?
        }

        if show_all || self.data.len() <= FORMATED_ELEMENT_MAX_COUNT {
            formatter.write_str("]")
        } else {
            formatter.write_str(", \u{2026}]")
        }
    }
}

fn fmt_tensor<V: Debug>(
    d4: usize,
    d3: usize,
    d2: usize,
    data: &Vec<V>,
    formatter: &mut Formatter<'_>,
) -> Result<(), Error> {
    let show_all = formatter.alternate();
    if show_all {
        display(data.iter(), (d4, d3, d2), formatter)?
    } else {
        display(
            data.iter().take(FORMATED_ELEMENT_MAX_COUNT),
            (d4, d3, d2),
            formatter,
        )?
    }

    if show_all || data.len() <= FORMATED_ELEMENT_MAX_COUNT {
        formatter.write_str("]")
    } else {
        formatter.write_str(", \u{2026}]")
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag, V: PartialEq + Debug> Debug
    for Tensor4<T1, T2, T3, T4, V>
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
        formatter.write_str("\u{3009}[")?;

        let d2 = self.d2.as_usize();
        let d3 = self.d3.as_usize() * d2;
        let d4 = self.d3.as_usize() * d3;
        fmt_tensor(d4, d3, d2, &self.data, formatter)
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag, T5: DimTag, V: PartialEq + Debug> Debug
    for Tensor5<T1, T2, T3, T4, T5, V>
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
        formatter.write_str("\u{3009}[")?;

        let d2 = self.d2.as_usize();
        let d3 = self.d3.as_usize() * d2;
        let d4 = self.d3.as_usize() * d3;
        fmt_tensor(d4, d3, d2, &self.data, formatter)
    }
}

impl<
        T1: DimTag,
        T2: DimTag,
        T3: DimTag,
        T4: DimTag,
        T5: DimTag,
        T6: DimTag,
        V: PartialEq + Debug,
    > Debug for Tensor6<T1, T2, T3, T4, T5, T6, V>
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
        formatter.write_str("\u{3009}[")?;

        let d2 = self.d2.as_usize();
        let d3 = self.d3.as_usize() * d2;
        let d4 = self.d3.as_usize() * d3;
        fmt_tensor(d4, d3, d2, &self.data, formatter)
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
        V: PartialEq + Debug,
    > Debug for Tensor7<T1, T2, T3, T4, T5, T6, T7, V>
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
        formatter.write_str("\u{3009}[")?;

        let d2 = self.d2.as_usize();
        let d3 = self.d3.as_usize() * d2;
        let d4 = self.d3.as_usize() * d3;
        fmt_tensor(d4, d3, d2, &self.data, formatter)
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
        V: PartialEq + Debug,
    > Debug for Tensor8<T1, T2, T3, T4, T5, T6, T7, T8, V>
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
        formatter.write_str("\u{3009}[")?;

        let d2 = self.d2.as_usize();
        let d3 = self.d3.as_usize() * d2;
        let d4 = self.d3.as_usize() * d3;
        fmt_tensor(d4, d3, d2, &self.data, formatter)
    }
}

#[inline]
fn make_index_3((i1, i2, i3): (usize, usize, usize), (d2, d3): (usize, usize)) -> usize {
    (i1 * d2 + i2) * d3 + i3
}

#[inline]
fn make_index_4(
    (i1, i2, i3, i4): (usize, usize, usize, usize),
    (d2, d3, d4): (usize, usize, usize),
) -> usize {
    make_index_3((i1, i2, i3), (d2, d3)) * d4 + i4
}

#[inline]
fn make_index_5(
    (i1, i2, i3, i4, i5): (usize, usize, usize, usize, usize),
    (d2, d3, d4, d5): (usize, usize, usize, usize),
) -> usize {
    make_index_4((i1, i2, i3, i4), (d2, d3, d4)) * d5 + i5
}

#[inline]
fn make_index_6(
    (i1, i2, i3, i4, i5, i6): (usize, usize, usize, usize, usize, usize),
    (d2, d3, d4, d5, d6): (usize, usize, usize, usize, usize),
) -> usize {
    make_index_5((i1, i2, i3, i4, i5), (d2, d3, d4, d5)) * d6 + i6
}

#[inline]
fn make_index_7(
    (i1, i2, i3, i4, i5, i6, i7): (usize, usize, usize, usize, usize, usize, usize),
    (d2, d3, d4, d5, d6, d7): (usize, usize, usize, usize, usize, usize),
) -> usize {
    make_index_6((i1, i2, i3, i4, i5, i6), (d2, d3, d4, d5, d6)) * d7 + i7
}

#[inline]
fn make_index_8(
    (i1, i2, i3, i4, i5, i6, i7, i8): (usize, usize, usize, usize, usize, usize, usize, usize),
    (d2, d3, d4, d5, d6, d7, d8): (usize, usize, usize, usize, usize, usize, usize),
) -> usize {
    make_index_7((i1, i2, i3, i4, i5, i6, i7), (d2, d3, d4, d5, d6, d7)) * d8 + i8
}

impl<V: PartialEq + Debug> Index<()> for Tensor0<V> {
    type Output = V;
    fn index(&self, _index: ()) -> &Self::Output {
        &self.data
    }
}

impl<V: PartialEq + Debug> IndexMut<()> for Tensor0<V> {
    fn index_mut(&mut self, _index: ()) -> &mut Self::Output {
        &mut self.data
    }
}

impl<T: DimTag, V: PartialEq + Debug> Index<(usize,)> for Tensor1<T, V> {
    type Output = V;
    fn index(&self, (i1,): (usize,)) -> &Self::Output {
        &self.data[i1]
    }
}

impl<T: DimTag, V: PartialEq + Debug> IndexMut<(usize,)> for Tensor1<T, V> {
    fn index_mut(&mut self, (i1,): (usize,)) -> &mut Self::Output {
        &mut self.data[i1]
    }
}

impl<T1: DimTag, T2: DimTag, V: PartialEq + Debug> Index<(usize, usize)> for Tensor2<T1, T2, V> {
    type Output = V;
    fn index(&self, (i1, i2): (usize, usize)) -> &Self::Output {
        if self.d1.as_usize() <= i1 || self.d2.as_usize() <= i2 {
            panic!("Invalid index")
        }
        let index = i1 * self.d2.as_usize() + i2;
        unsafe { &self.data.get_unchecked(index) }
    }
}

impl<T1: DimTag, T2: DimTag, V: PartialEq + Debug> IndexMut<(usize, usize)> for Tensor2<T1, T2, V> {
    fn index_mut(&mut self, (i1, i2): (usize, usize)) -> &mut Self::Output {
        if self.d1.as_usize() <= i1 || self.d2.as_usize() <= i2 {
            panic!("Invalid index")
        }
        let index = i1 * self.d2.as_usize() + i2;
        unsafe { self.data.get_unchecked_mut(index) }
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, V: PartialEq + Debug> Index<(usize, usize, usize)>
    for Tensor3<T1, T2, T3, V>
{
    type Output = V;
    fn index(&self, (i1, i2, i3): (usize, usize, usize)) -> &Self::Output {
        if self.d1.as_usize() <= i1 || self.d2.as_usize() <= i2 || self.d3.as_usize() <= i3 {
            panic!("Invalid index")
        }
        let index = make_index_3((i1, i2, i3), (self.d2.as_usize(), self.d3.as_usize()));
        unsafe { &self.data.get_unchecked(index) }
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, V: PartialEq + Debug> IndexMut<(usize, usize, usize)>
    for Tensor3<T1, T2, T3, V>
{
    fn index_mut(&mut self, (i1, i2, i3): (usize, usize, usize)) -> &mut Self::Output {
        if self.d1.as_usize() <= i1 || self.d2.as_usize() <= i2 || self.d3.as_usize() <= i3 {
            panic!("Invalid index")
        }
        let index = make_index_3((i1, i2, i3), (self.d2.as_usize(), self.d3.as_usize()));
        unsafe { self.data.get_unchecked_mut(index) }
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag, V: PartialEq + Debug>
    Index<(usize, usize, usize, usize)> for Tensor4<T1, T2, T3, T4, V>
{
    type Output = V;
    fn index(&self, (i1, i2, i3, i4): (usize, usize, usize, usize)) -> &Self::Output {
        if self.d1.as_usize() <= i1
            || self.d2.as_usize() <= i2
            || self.d3.as_usize() <= i3
            || self.d4.as_usize() <= i4
        {
            panic!("Invalid index")
        }
        let index = make_index_4(
            (i1, i2, i3, i4),
            (self.d2.as_usize(), self.d3.as_usize(), self.d4.as_usize()),
        );
        unsafe { &self.data.get_unchecked(index) }
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag, V: PartialEq + Debug>
    IndexMut<(usize, usize, usize, usize)> for Tensor4<T1, T2, T3, T4, V>
{
    fn index_mut(&mut self, (i1, i2, i3, i4): (usize, usize, usize, usize)) -> &mut Self::Output {
        if self.d1.as_usize() <= i1
            || self.d2.as_usize() <= i2
            || self.d3.as_usize() <= i3
            || self.d4.as_usize() <= i4
        {
            panic!("Invalid index")
        }
        let index = make_index_4(
            (i1, i2, i3, i4),
            (self.d2.as_usize(), self.d3.as_usize(), self.d4.as_usize()),
        );
        unsafe { self.data.get_unchecked_mut(index) }
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag, T5: DimTag, V: PartialEq + Debug>
    Index<(usize, usize, usize, usize, usize)> for Tensor5<T1, T2, T3, T4, T5, V>
{
    type Output = V;
    fn index(&self, (i1, i2, i3, i4, i5): (usize, usize, usize, usize, usize)) -> &Self::Output {
        if self.d1.as_usize() <= i1
            || self.d2.as_usize() <= i2
            || self.d3.as_usize() <= i3
            || self.d4.as_usize() <= i4
            || self.d5.as_usize() <= i5
        {
            panic!("Invalid index")
        }
        let index = make_index_5(
            (i1, i2, i3, i4, i5),
            (
                self.d2.as_usize(),
                self.d3.as_usize(),
                self.d4.as_usize(),
                self.d5.as_usize(),
            ),
        );
        unsafe { &self.data.get_unchecked(index) }
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag, T5: DimTag, V: PartialEq + Debug>
    IndexMut<(usize, usize, usize, usize, usize)> for Tensor5<T1, T2, T3, T4, T5, V>
{
    fn index_mut(
        &mut self,
        (i1, i2, i3, i4, i5): (usize, usize, usize, usize, usize),
    ) -> &mut Self::Output {
        if self.d1.as_usize() <= i1
            || self.d2.as_usize() <= i2
            || self.d3.as_usize() <= i3
            || self.d4.as_usize() <= i4
            || self.d5.as_usize() <= i5
        {
            panic!("Invalid index")
        }
        let index = make_index_5(
            (i1, i2, i3, i4, i5),
            (
                self.d2.as_usize(),
                self.d3.as_usize(),
                self.d4.as_usize(),
                self.d5.as_usize(),
            ),
        );
        unsafe { self.data.get_unchecked_mut(index) }
    }
}

impl<
        T1: DimTag,
        T2: DimTag,
        T3: DimTag,
        T4: DimTag,
        T5: DimTag,
        T6: DimTag,
        V: PartialEq + Debug,
    > Index<(usize, usize, usize, usize, usize, usize)> for Tensor6<T1, T2, T3, T4, T5, T6, V>
{
    type Output = V;
    fn index(
        &self,
        (i1, i2, i3, i4, i5, i6): (usize, usize, usize, usize, usize, usize),
    ) -> &Self::Output {
        if self.d1.as_usize() <= i1
            || self.d2.as_usize() <= i2
            || self.d3.as_usize() <= i3
            || self.d4.as_usize() <= i4
            || self.d5.as_usize() <= i5
            || self.d6.as_usize() <= i6
        {
            panic!("Invalid index")
        }
        let index = make_index_6(
            (i1, i2, i3, i4, i5, i6),
            (
                self.d2.as_usize(),
                self.d3.as_usize(),
                self.d4.as_usize(),
                self.d5.as_usize(),
                self.d6.as_usize(),
            ),
        );
        unsafe { &self.data.get_unchecked(index) }
    }
}

impl<
        T1: DimTag,
        T2: DimTag,
        T3: DimTag,
        T4: DimTag,
        T5: DimTag,
        T6: DimTag,
        V: PartialEq + Debug,
    > IndexMut<(usize, usize, usize, usize, usize, usize)> for Tensor6<T1, T2, T3, T4, T5, T6, V>
{
    fn index_mut(
        &mut self,
        (i1, i2, i3, i4, i5, i6): (usize, usize, usize, usize, usize, usize),
    ) -> &mut Self::Output {
        if self.d1.as_usize() <= i1
            || self.d2.as_usize() <= i2
            || self.d3.as_usize() <= i3
            || self.d4.as_usize() <= i4
            || self.d5.as_usize() <= i5
            || self.d6.as_usize() <= i6
        {
            panic!("Invalid index")
        }
        let index = make_index_6(
            (i1, i2, i3, i4, i5, i6),
            (
                self.d2.as_usize(),
                self.d3.as_usize(),
                self.d4.as_usize(),
                self.d5.as_usize(),
                self.d6.as_usize(),
            ),
        );
        unsafe { self.data.get_unchecked_mut(index) }
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
        V: PartialEq + Debug,
    > Index<(usize, usize, usize, usize, usize, usize, usize)>
    for Tensor7<T1, T2, T3, T4, T5, T6, T7, V>
{
    type Output = V;
    fn index(
        &self,
        (i1, i2, i3, i4, i5, i6, i7): (usize, usize, usize, usize, usize, usize, usize),
    ) -> &Self::Output {
        if self.d1.as_usize() <= i1
            || self.d2.as_usize() <= i2
            || self.d3.as_usize() <= i3
            || self.d4.as_usize() <= i4
            || self.d5.as_usize() <= i5
            || self.d6.as_usize() <= i6
            || self.d7.as_usize() <= i7
        {
            panic!("Invalid index")
        }
        let index = make_index_7(
            (i1, i2, i3, i4, i5, i6, i7),
            (
                self.d2.as_usize(),
                self.d3.as_usize(),
                self.d4.as_usize(),
                self.d5.as_usize(),
                self.d6.as_usize(),
                self.d7.as_usize(),
            ),
        );
        unsafe { &self.data.get_unchecked(index) }
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
        V: PartialEq + Debug,
    > IndexMut<(usize, usize, usize, usize, usize, usize, usize)>
    for Tensor7<T1, T2, T3, T4, T5, T6, T7, V>
{
    fn index_mut(
        &mut self,
        (i1, i2, i3, i4, i5, i6, i7): (usize, usize, usize, usize, usize, usize, usize),
    ) -> &mut Self::Output {
        if self.d1.as_usize() <= i1
            || self.d2.as_usize() <= i2
            || self.d3.as_usize() <= i3
            || self.d4.as_usize() <= i4
            || self.d5.as_usize() <= i5
            || self.d6.as_usize() <= i6
            || self.d7.as_usize() <= i7
        {
            panic!("Invalid index")
        }
        let index = make_index_7(
            (i1, i2, i3, i4, i5, i6, i7),
            (
                self.d2.as_usize(),
                self.d3.as_usize(),
                self.d4.as_usize(),
                self.d5.as_usize(),
                self.d6.as_usize(),
                self.d7.as_usize(),
            ),
        );
        unsafe { self.data.get_unchecked_mut(index) }
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
        V: PartialEq + Debug,
    > Index<(usize, usize, usize, usize, usize, usize, usize, usize)>
    for Tensor8<T1, T2, T3, T4, T5, T6, T7, T8, V>
{
    type Output = V;
    fn index(
        &self,
        (i1, i2, i3, i4, i5, i6, i7, i8): (usize, usize, usize, usize, usize, usize, usize, usize),
    ) -> &Self::Output {
        if self.d1.as_usize() <= i1
            || self.d2.as_usize() <= i2
            || self.d3.as_usize() <= i3
            || self.d4.as_usize() <= i4
            || self.d5.as_usize() <= i5
            || self.d6.as_usize() <= i6
            || self.d7.as_usize() <= i7
            || self.d8.as_usize() <= i8
        {
            panic!("Invalid index")
        }
        let index = make_index_8(
            (i1, i2, i3, i4, i5, i6, i7, i8),
            (
                self.d2.as_usize(),
                self.d3.as_usize(),
                self.d4.as_usize(),
                self.d5.as_usize(),
                self.d6.as_usize(),
                self.d7.as_usize(),
                self.d8.as_usize(),
            ),
        );
        unsafe { &self.data.get_unchecked(index) }
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
        V: PartialEq + Debug,
    > IndexMut<(usize, usize, usize, usize, usize, usize, usize, usize)>
    for Tensor8<T1, T2, T3, T4, T5, T6, T7, T8, V>
{
    fn index_mut(
        &mut self,
        (i1, i2, i3, i4, i5, i6, i7, i8): (usize, usize, usize, usize, usize, usize, usize, usize),
    ) -> &mut Self::Output {
        if self.d1.as_usize() <= i1
            || self.d2.as_usize() <= i2
            || self.d3.as_usize() <= i3
            || self.d4.as_usize() <= i4
            || self.d5.as_usize() <= i5
            || self.d6.as_usize() <= i6
            || self.d7.as_usize() <= i7
            || self.d8.as_usize() <= i8
        {
            panic!("Invalid index")
        }
        let index = make_index_8(
            (i1, i2, i3, i4, i5, i6, i7, i8),
            (
                self.d2.as_usize(),
                self.d3.as_usize(),
                self.d4.as_usize(),
                self.d5.as_usize(),
                self.d6.as_usize(),
                self.d7.as_usize(),
                self.d8.as_usize(),
            ),
        );
        unsafe { self.data.get_unchecked_mut(index) }
    }
}

impl<V: PartialEq + Debug> Tensor0<V> {
    pub fn new(value: V) -> Self {
        Tensor0::<V> { data: value }
    }
    pub fn get(&self) -> &V {
        &self.data
    }
}

impl<T: DimTag, V: PartialEq + Debug> Tensor1<T, V> {
    pub unsafe fn get_unchecked(&self, i: usize) -> &V {
        self.data.get_unchecked(i)
    }

    pub fn from_vector_with_dim(d: Dim<T>, mut values: Vec<V>) -> Self {
        if values.len() < d.as_usize() {
            panic!("Array too short")
        }
        values.truncate(d.as_usize());
        Tensor1::<T, V> {
            phantom: PhantomData {},
            data: values,
        }
    }

    pub fn try_cast<DPrime: DimTag>(self, d: Dim<DPrime>) -> Result<Tensor1<DPrime, V>, Self> {
        if self.data.len() == d.as_usize() {
            Ok(Tensor1::<DPrime, V> {
                phantom: PhantomData {},
                data: self.data,
            })
        } else {
            Err(self)
        }
    }
}

impl<T1: DimTag, T2: DimTag, V: PartialEq + Debug> Tensor2<T1, T2, V> {
    pub unsafe fn get_unchecked(&self, i1: usize, i2: usize) -> &V {
        let index = i1 * self.d2.as_usize() + i2;
        self.data.get_unchecked(index)
    }

    pub fn try_cast<E1: DimTag, E2: DimTag>(
        self,
        d1: Dim<E1>,
        d2: Dim<E2>,
    ) -> Result<Tensor2<E1, E2, V>, Self> {
        if self.d1 == d1 && self.d2 == d2 {
            Ok(Tensor2::<E1, E2, V> {
                d1,
                d2,
                data: self.data,
            })
        } else {
            Err(self)
        }
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, V: PartialEq + Debug> Tensor3<T1, T2, T3, V> {
    pub unsafe fn get_unchecked(&self, i1: usize, i2: usize, i3: usize) -> &V {
        let index = make_index_3((i1, i2, i3), (self.d2.as_usize(), self.d3.as_usize()));
        self.data.get_unchecked(index)
    }

    pub fn try_cast<E1: DimTag, E2: DimTag, E3: DimTag>(
        self,
        d1: Dim<E1>,
        d2: Dim<E2>,
        d3: Dim<E3>,
    ) -> Result<Tensor3<E1, E2, E3, V>, Self> {
        if self.d1 == d1 && self.d2 == d2 && self.d3 == d3 {
            Ok(Tensor3::<E1, E2, E3, V> {
                d1,
                d2,
                d3,
                data: self.data,
            })
        } else {
            Err(self)
        }
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag, V: PartialEq + Debug>
    Tensor4<T1, T2, T3, T4, V>
{
    pub unsafe fn get_unchecked(&self, i1: usize, i2: usize, i3: usize, i4: usize) -> &V {
        let index = make_index_4(
            (i1, i2, i3, i4),
            (self.d2.as_usize(), self.d3.as_usize(), self.d4.as_usize()),
        );
        self.data.get_unchecked(index)
    }

    pub fn try_cast<E1: DimTag, E2: DimTag, E3: DimTag, E4: DimTag>(
        self,
        d1: Dim<E1>,
        d2: Dim<E2>,
        d3: Dim<E3>,
        d4: Dim<E4>,
    ) -> Result<Tensor4<E1, E2, E3, E4, V>, Self> {
        if self.d1 == d1 && self.d2 == d2 && self.d3 == d3 && self.d4 == d4 {
            Ok(Tensor4::<E1, E2, E3, E4, V> {
                d1,
                d2,
                d3,
                d4,
                data: self.data,
            })
        } else {
            Err(self)
        }
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag, T5: DimTag, V: PartialEq + Debug>
    Tensor5<T1, T2, T3, T4, T5, V>
{
    pub unsafe fn get_unchecked(
        &self,
        i1: usize,
        i2: usize,
        i3: usize,
        i4: usize,
        i5: usize,
    ) -> &V {
        let index = make_index_5(
            (i1, i2, i3, i4, i5),
            (
                self.d2.as_usize(),
                self.d3.as_usize(),
                self.d4.as_usize(),
                self.d5.as_usize(),
            ),
        );
        self.data.get_unchecked(index)
    }

    pub fn try_cast<E1: DimTag, E2: DimTag, E3: DimTag, E4: DimTag, E5: DimTag>(
        self,
        d1: Dim<E1>,
        d2: Dim<E2>,
        d3: Dim<E3>,
        d4: Dim<E4>,
        d5: Dim<E5>,
    ) -> Result<Tensor5<E1, E2, E3, E4, E5, V>, Self> {
        if self.d1 == d1 && self.d2 == d2 && self.d3 == d3 && self.d4 == d4 && self.d5 == d5 {
            Ok(Tensor5::<E1, E2, E3, E4, E5, V> {
                d1,
                d2,
                d3,
                d4,
                d5,
                data: self.data,
            })
        } else {
            Err(self)
        }
    }
}

impl<
        T1: DimTag,
        T2: DimTag,
        T3: DimTag,
        T4: DimTag,
        T5: DimTag,
        T6: DimTag,
        V: PartialEq + Debug,
    > Tensor6<T1, T2, T3, T4, T5, T6, V>
{
    pub unsafe fn get_unchecked(
        &self,
        i1: usize,
        i2: usize,
        i3: usize,
        i4: usize,
        i5: usize,
        i6: usize,
    ) -> &V {
        let index = make_index_6(
            (i1, i2, i3, i4, i5, i6),
            (
                self.d2.as_usize(),
                self.d3.as_usize(),
                self.d4.as_usize(),
                self.d5.as_usize(),
                self.d6.as_usize(),
            ),
        );
        self.data.get_unchecked(index)
    }

    pub fn try_cast<E1: DimTag, E2: DimTag, E3: DimTag, E4: DimTag, E5: DimTag, E6: DimTag>(
        self,
        d1: Dim<E1>,
        d2: Dim<E2>,
        d3: Dim<E3>,
        d4: Dim<E4>,
        d5: Dim<E5>,
        d6: Dim<E6>,
    ) -> Result<Tensor6<E1, E2, E3, E4, E5, E6, V>, Self> {
        if self.d1 == d1
            && self.d2 == d2
            && self.d3 == d3
            && self.d4 == d4
            && self.d5 == d5
            && self.d6 == d6
        {
            Ok(Tensor6::<E1, E2, E3, E4, E5, E6, V> {
                d1,
                d2,
                d3,
                d4,
                d5,
                d6,
                data: self.data,
            })
        } else {
            Err(self)
        }
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
        V: PartialEq + Debug,
    > Tensor7<T1, T2, T3, T4, T5, T6, T7, V>
{
    pub unsafe fn get_unchecked(
        &self,
        i1: usize,
        i2: usize,
        i3: usize,
        i4: usize,
        i5: usize,
        i6: usize,
        i7: usize,
    ) -> &V {
        let index = make_index_7(
            (i1, i2, i3, i4, i5, i6, i7),
            (
                self.d2.as_usize(),
                self.d3.as_usize(),
                self.d4.as_usize(),
                self.d5.as_usize(),
                self.d6.as_usize(),
                self.d7.as_usize(),
            ),
        );
        self.data.get_unchecked(index)
    }

    pub fn try_cast<
        E1: DimTag,
        E2: DimTag,
        E3: DimTag,
        E4: DimTag,
        E5: DimTag,
        E6: DimTag,
        E7: DimTag,
    >(
        self,
        d1: Dim<E1>,
        d2: Dim<E2>,
        d3: Dim<E3>,
        d4: Dim<E4>,
        d5: Dim<E5>,
        d6: Dim<E6>,
        d7: Dim<E7>,
    ) -> Result<Tensor7<E1, E2, E3, E4, E5, E6, E7, V>, Self> {
        if self.d1 == d1
            && self.d2 == d2
            && self.d3 == d3
            && self.d4 == d4
            && self.d5 == d5
            && self.d6 == d6
            && self.d7 == d7
        {
            Ok(Tensor7::<E1, E2, E3, E4, E5, E6, E7, V> {
                d1,
                d2,
                d3,
                d4,
                d5,
                d6,
                d7,
                data: self.data,
            })
        } else {
            Err(self)
        }
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
        V: PartialEq + Debug,
    > Tensor8<T1, T2, T3, T4, T5, T6, T7, T8, V>
{
    pub unsafe fn get_unchecked(
        &self,
        i1: usize,
        i2: usize,
        i3: usize,
        i4: usize,
        i5: usize,
        i6: usize,
        i7: usize,
        i8: usize,
    ) -> &V {
        let index = make_index_8(
            (i1, i2, i3, i4, i5, i6, i7, i8),
            (
                self.d2.as_usize(),
                self.d3.as_usize(),
                self.d4.as_usize(),
                self.d5.as_usize(),
                self.d6.as_usize(),
                self.d7.as_usize(),
                self.d8.as_usize(),
            ),
        );
        self.data.get_unchecked(index)
    }

    pub fn try_cast<
        E1: DimTag,
        E2: DimTag,
        E3: DimTag,
        E4: DimTag,
        E5: DimTag,
        E6: DimTag,
        E7: DimTag,
        E8: DimTag,
    >(
        self,
        d1: Dim<E1>,
        d2: Dim<E2>,
        d3: Dim<E3>,
        d4: Dim<E4>,
        d5: Dim<E5>,
        d6: Dim<E6>,
        d7: Dim<E7>,
        d8: Dim<E8>,
    ) -> Result<Tensor8<E1, E2, E3, E4, E5, E6, E7, E8, V>, Self> {
        if self.d1 == d1
            && self.d2 == d2
            && self.d3 == d3
            && self.d4 == d4
            && self.d5 == d5
            && self.d6 == d6
            && self.d7 == d7
            && self.d8 == d8
        {
            Ok(Tensor8::<E1, E2, E3, E4, E5, E6, E7, E8, V> {
                d1,
                d2,
                d3,
                d4,
                d5,
                d6,
                d7,
                d8,
                data: self.data,
            })
        } else {
            Err(self)
        }
    }
}
