//! The mathematical word tensor is used here more broadly to describe
//! arrays with multiple indexes containing copiable data.

use super::dimensions::*;
use std::fmt::{Debug, Error, Formatter};
use std::marker::{Copy, PhantomData};
use std::ops::{Index, IndexMut};

/// A general trait shared by all tensor types.
pub trait Tensor: PartialEq + Index<Self::MultiIndex> + IndexMut<Self::MultiIndex> + Debug {
    /// The type of elements stored in the tensor.
    type Element: PartialEq + Debug;
    /// The type of the dimensions tuple
    type Dimensions: Copy + Eq + Debug;
    // the type of a multi-index to retrieve an element
    type MultiIndex: Copy + Eq + Debug;
    /// The number of dimensions, also corresponding
    /// to the number of indexes.
    const RANK: u16;
    /// The total number of coordinates
    fn count(&self) -> u64;
    /// The tuple of dimensions
    fn dims(&self) -> Self::Dimensions;
    /// Update some elements of the tensor
    fn update(&mut self, updater: impl FnMut(Self::MultiIndex, &mut Self::Element));
}

#[derive(PartialEq)]
pub struct Tensor0<V: PartialEq + Debug> {
    pub(crate) data: V,
}

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
    pub(crate) phantom: PhantomData<(T1, T2, T3, T4)>,
    #[allow(dead_code)]
    data: Vec<V>,
}

pub type Scalar<V> = Tensor0<V>;
pub type Vector<T, V> = Tensor1<T, V>;
pub type Matrix<T1, T2, V> = Tensor2<T1, T2, V>;

pub type StaticVector<const N: usize, V> = Tensor1<StaticDimTag<N>, V>;
pub type StaticMatrix<const N1: usize, const N2: usize, V> =
    Tensor2<StaticDimTag<N1>, StaticDimTag<N2>, V>;

impl<V: PartialEq + Debug> Tensor for Tensor0<V> {
    type Element = V;
    type Dimensions = ();
    type MultiIndex = ();
    const RANK: u16 = 0;
    fn count(&self) -> u64 {
        1
    }
    fn dims(&self) -> Self::Dimensions {}
    fn update(&mut self, mut updater: impl FnMut((), &mut V)) {
        updater((), &mut self.data)
    }
}

impl<T: DimTag, V: PartialEq + Debug> Tensor for Tensor1<T, V> {
    type Element = V;
    type Dimensions = (Dim<T>,);
    type MultiIndex = (usize,);
    const RANK: u16 = 1;
    fn count(&self) -> u64 {
        self.data.len() as u64
    }
    fn dims(&self) -> Self::Dimensions {
        let n = self.data.len();
        unsafe { (Dim::<T>::unsafe_new(n),) }
    }
    fn update(&mut self, mut updater: impl FnMut((usize,), &mut V)) {
        for (i, element) in self.data.iter_mut().enumerate() {
            updater((i,), element)
        }
    }
}

impl<T1: DimTag, T2: DimTag, V: PartialEq + Debug> Tensor for Tensor2<T1, T2, V> {
    type Element = V;
    type Dimensions = (Dim<T1>, Dim<T2>);
    type MultiIndex = (usize, usize);
    const RANK: u16 = 2;
    fn count(&self) -> u64 {
        self.data.len() as u64
    }
    fn dims(&self) -> Self::Dimensions {
        (self.d1, self.d2)
    }
    fn update(&mut self, mut updater: impl FnMut((usize, usize), &mut V)) {
        let d2 = self.d2.as_usize();
        for (i, element) in self.data.iter_mut().enumerate() {
            let i1 = i / d2;
            let i2 = i % d2;
            updater((i1, i2), element)
        }
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, V: PartialEq + Debug> Tensor for Tensor3<T1, T2, T3, V> {
    type Element = V;
    type Dimensions = (Dim<T1>, Dim<T2>, Dim<T3>);
    type MultiIndex = (usize, usize, usize);
    const RANK: u16 = 3;
    fn count(&self) -> u64 {
        self.data.len() as u64
    }
    fn dims(&self) -> Self::Dimensions {
        (self.d1, self.d2, self.d3)
    }
    fn update(&mut self, mut updater: impl FnMut((usize, usize, usize), &mut V)) {
        let d2 = self.d2.as_usize();
        let d3 = self.d3.as_usize();
        for (i, element) in self.data.iter_mut().enumerate() {
            let i3 = i % d3;
            let i12 = i / d3;
            let i1 = i12 / d2;
            let i2 = i12 % d2;
            updater((i1, i2, i3), element)
        }
    }
}

impl<V: Clone + PartialEq + Debug> Clone for Tensor0<V> {
    fn clone(&self) -> Self {
        Tensor0 {
            data: self.data.clone(),
        }
    }
}

impl<V: Copy + PartialEq + Debug> Copy for Tensor0<V> {}

impl<T: DimTag, V: Clone + PartialEq + Debug> Clone for Tensor1<T, V> {
    fn clone(&self) -> Self {
        Tensor1 {
            phantom: PhantomData {},
            data: self.data.clone(),
        }
    }
}

impl<T1: DimTag, T2: DimTag, V: Clone + PartialEq + Debug> Clone for Tensor2<T1, T2, V> {
    fn clone(&self) -> Self {
        Tensor2 {
            d1: self.d1,
            d2: self.d2,
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
    fn index(&self, index: (usize,)) -> &Self::Output {
        &self.data[index.0]
    }
}

impl<T: DimTag, V: PartialEq + Debug> IndexMut<(usize,)> for Tensor1<T, V> {
    fn index_mut(&mut self, index: (usize,)) -> &mut Self::Output {
        &mut self.data[index.0]
    }
}

impl<T1: DimTag, T2: DimTag, V: PartialEq + Debug> Index<(usize, usize)> for Tensor2<T1, T2, V> {
    type Output = V;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        if self.d1.as_usize() <= index.0 || self.d2.as_usize() <= index.1 {
            panic!("Invalid index")
        }
        let index = index.0 * self.d2.as_usize() + index.1;
        unsafe { &self.data.get_unchecked(index) }
    }
}

impl<T1: DimTag, T2: DimTag, V: PartialEq + Debug> IndexMut<(usize, usize)> for Tensor2<T1, T2, V> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        if self.d1.as_usize() <= index.0 || self.d2.as_usize() <= index.1 {
            panic!("Invalid index")
        }
        let index = index.0 * self.d2.as_usize() + index.1;
        unsafe { self.data.get_unchecked_mut(index) }
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, V: PartialEq + Debug> Index<(usize, usize, usize)>
    for Tensor3<T1, T2, T3, V>
{
    type Output = V;
    fn index(&self, index: (usize, usize, usize)) -> &Self::Output {
        if self.d1.as_usize() <= index.0
            || self.d2.as_usize() <= index.1
            || self.d3.as_usize() <= index.2
        {
            panic!("Invalid index")
        }
        let index = (index.0 * self.d2.as_usize() + index.1) * self.d3.as_usize() + index.2;
        unsafe { &self.data.get_unchecked(index) }
    }
}

impl<T1: DimTag, T2: DimTag, T3: DimTag, V: PartialEq + Debug> IndexMut<(usize, usize, usize)>
    for Tensor3<T1, T2, T3, V>
{
    fn index_mut(&mut self, index: (usize, usize, usize)) -> &mut Self::Output {
        if self.d1.as_usize() <= index.0
            || self.d2.as_usize() <= index.1
            || self.d3.as_usize() <= index.2
        {
            panic!("Invalid index")
        }
        let index = (index.0 * self.d2.as_usize() + index.1) * self.d3.as_usize() + index.2;
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
}

impl<T: DimTag, V: PartialEq + Debug> Tensor1<T, V> {
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
        self.data.get_unchecked(i1 * self.d2.as_usize() + i2)
    }

    pub fn try_cast<D1Prime: DimTag, D2Prime: DimTag>(
        self,
        d1: Dim<D1Prime>,
        d2: Dim<D2Prime>,
    ) -> Result<Tensor2<D1Prime, D2Prime, V>, Self> {
        if self.d1 == d1 && self.d2 == d2 {
            Ok(Tensor2::<D1Prime, D2Prime, V> {
                d1,
                d2,
                data: self.data,
            })
        } else {
            Err(self)
        }
    }
}
