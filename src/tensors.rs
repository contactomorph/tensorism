//! The mathematical word tensor is used here more broadly to describe
//! arrays with multiple indexes containing copiable data.

use crate::shapes::{Shape, Tensor1Shape, Tensor2Shape, Tensor3Shape};

use super::dimensions::*;
use std::fmt::{Debug, Error, Formatter};
use std::marker::{Copy, PhantomData};
use std::ops::Index;

/// A general trait shared by all tensor types.
pub trait Tensor: PartialEq + Index<Self::MultiIndex> + Debug {
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
    data: V,
}

#[derive(PartialEq)]
pub struct Tensor1<T: DimTag, V: PartialEq + Debug> {
    phantom: PhantomData<T>,
    data: Vec<V>,
}

#[derive(PartialEq)]
pub struct Tensor2<T1: DimTag, T2: DimTag, V: PartialEq + Debug> {
    d1: Dim<T1>,
    d2: Dim<T2>,
    data: Vec<V>,
}

#[derive(PartialEq)]
pub struct Tensor3<T1: DimTag, T2: DimTag, T3: DimTag, V: PartialEq + Debug> {
    d1: Dim<T1>,
    d2: Dim<T2>,
    d3: Dim<T3>,
    data: Vec<V>,
}

#[derive(PartialEq)]
pub struct Tensor4<T1: DimTag, T2: DimTag, T3: DimTag, T4: DimTag, V: PartialEq + Debug> {
    phantom: PhantomData<(T1, T2, T3, T4)>,
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

impl<V: Clone + Eq + Debug> Clone for Tensor0<V> {
    fn clone(&self) -> Self {
        Tensor0 {
            data: self.data.clone(),
        }
    }
}

impl<V: Copy + Eq + Debug> Copy for Tensor0<V> {}

impl<T: DimTag, V: Clone + Eq + Debug> Clone for Tensor1<T, V> {
    fn clone(&self) -> Self {
        Tensor1 {
            phantom: PhantomData {},
            data: self.data.clone(),
        }
    }
}

impl<T1: DimTag, T2: DimTag, V: Clone + Eq + Debug> Clone for Tensor2<T1, T2, V> {
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

impl<T: DimTag, V: PartialEq + Debug> Index<(usize,)> for Tensor1<T, V> {
    type Output = V;
    fn index(&self, index: (usize,)) -> &Self::Output {
        &self.data[index.0]
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

impl<V: PartialEq + Debug> Tensor0<V> {
    pub fn new(value: V) -> Self {
        Tensor0::<V> { data: value }
    }
    pub fn get(&self) -> &V {
        &self.data
    }
}

impl<T: DimTag, V: PartialEq + Debug> Tensor1<T, V> {
    pub fn get(&self, i: usize) -> &V {
        &self.data[i]
    }

    pub unsafe fn get_unchecked(&self, i: usize) -> &V {
        self.data.get_unchecked(i)
    }

    pub fn set(&mut self, i: usize, value: V) {
        self.data[i] = value
    }

    pub unsafe fn set_unchecked(&mut self, _i: usize, _value: V) {
        //self.data.get_unchecked_mut(index) = value
        panic!("")
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
    pub fn set___<VInput: Eq + Debug>(
        &mut self,
        input: &Tensor1<T, VInput>,
        mut f: impl FnMut(&VInput) -> V,
    ) {
        let ptr_o = self.data.as_mut_ptr();
        let ptr = input.data.as_ptr();
        for i in 0usize..self.data.len() {
            unsafe {
                let ptr = ptr.add(i);
                *ptr_o.add(i) = f(&*ptr);
            }
        }
    }
    pub fn set_from_2<V1: Eq + Debug, V2: Eq + Debug>(
        &mut self,
        input1: &Tensor1<T, V1>,
        input2: &Tensor1<T, V2>,
        mut f: impl FnMut(&V1, &V2) -> V,
    ) {
        let ptr_o = self.data.as_mut_ptr();
        let ptr1 = input1.data.as_ptr();
        let ptr2 = input2.data.as_ptr();
        for i in 0usize..self.data.len() {
            unsafe {
                let ptr1 = ptr1.add(i);
                let ptr2 = ptr2.add(i);
                *ptr_o.add(i) = f(&*ptr1, &*ptr2);
            }
        }
    }

    pub fn map<W: Eq + Debug>(&self, mut _f: impl FnMut(&V) -> W) -> Tensor1<T, W> {
        panic!("not impl")
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
    pub unsafe fn get(&self, i1: usize, i2: usize) -> &V {
        if i1 >= self.d1.as_usize() || i2 >= self.d2.as_usize() {
            panic!("Incorrect index")
        }
        self.get_unchecked(i1, i2)
    }

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

#[doc(hidden)]
pub unsafe fn from_array_to_tensor1_unchecked<T: DimTag, V: Copy + Eq + Debug>(
    values: Vec<V>,
) -> Tensor1<T, V> {
    Tensor1::<T, V> {
        phantom: PhantomData {},
        data: values,
    }
}

pub fn from_array<V: Copy + Eq + Debug, const N: usize>(
    array: &[V; N],
) -> Tensor1<StaticDimTag<N>, V> {
    Tensor1::<StaticDimTag<N>, V> {
        phantom: PhantomData {},
        data: array.to_vec(),
    }
}

#[macro_export]
macro_rules! from_array_to_tensor1 {
    ($values:expr) => {{
        enum UnnamedTag {}
        lazy_static::lazy_static! {
            static ref THUMBPRINT: u16 = generate_thumbprint();
        }
        impl DimTag for UnnamedTag {
            fn get_thumbprint() -> u16 {
                *THUMBPRINT
            }
        }
        unsafe { from_array_to_tensor1_unchecked::<UnnamedTag, _>($values) }
    }};
}

pub struct TensorPreparation<T: Tensor> {
    expected_size: usize,
    dims: T::Dimensions,
    data: Vec<T::Element>,
    generator: fn(T::Dimensions, Vec<T::Element>) -> T,
}

impl<T: Tensor> TensorPreparation<T> {
    pub fn count_set_elements(&self) -> usize {
        self.data.len()
    }
    pub fn count_unset_elements(&self) -> usize {
        self.expected_size - self.data.len()
    }
    pub fn append(mut self, values: &mut Vec<T::Element>) -> Self {
        self.data.append(values);
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
    pub fn generate(self) -> T {
        if self.data.len() != self.expected_size {
            panic!("Invalid size")
        }
        (self.generator)(self.dims, self.data)
    }
    pub fn try_generate(self) -> Result<T, Self> {
        if self.data.len() != self.expected_size {
            Err(self)
        } else {
            Ok((self.generator)(self.dims, self.data))
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
    type MultiIndex = usize;

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
            data.push(f(i))
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
            dims: (self.d,),
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
            dims: (self.d1, self.d2),
            generator: |dims, data| Self::Tensor {
                d1: dims.0,
                d2: dims.1,
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
            dims: (self.d1, self.d2, self.d3),
            generator: |dims, data| Self::Tensor {
                d1: dims.0,
                d2: dims.1,
                d3: dims.2,
                data,
            },
        }
    }
}
